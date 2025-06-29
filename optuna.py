import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os

class PolyclinicConstants:
    """
    Константы, используемые в среде симуляции поликлиники.
    """
    TIME_STEP_MINUTES = 5  # Шаг времени в минутах
    NUM_DOCTORS = 5  # Количество врачей
    WORK_START_HOUR = 8  # Время начала работы (часы)
    WORK_END_HOUR = 21  # Время окончания работы (часы)

    # Продолжительность приемов в минутах
    CONSULTATION_DURATION_MINUTES = 10
    EMERGENCY_DURATION_MINUTES = 20
    PROCEDURE_DURATION_MINUTES = 15

    MAX_WAITING_ROOM_CAPACITY = 30 # Максимальное количество пациентов в зале ожидания (суммарно)

    # Типы пациентов и их продолжительность в шагах времени
    PATIENT_TYPES = {
        "consultation": {
            "id": 0,
            "duration_steps": CONSULTATION_DURATION_MINUTES // TIME_STEP_MINUTES
        },
        "emergency": {
            "id": 1,
            "duration_steps": EMERGENCY_DURATION_MINUTES // TIME_STEP_MINUTES
        },
        "procedure": {
            "id": 2,
            "duration_steps": PROCEDURE_DURATION_MINUTES // TIME_STEP_MINUTES
        }
    }
    NUM_PATIENT_TYPES = len(PATIENT_TYPES)

    # Общее количество временных шагов в рабочий день
    TOTAL_WORK_MINUTES = (WORK_END_HOUR - WORK_START_HOUR) * 60
    MAX_TIME_STEPS_IN_DAY = TOTAL_WORK_MINUTES // TIME_STEP_MINUTES

    # Индексы в массиве ожидания для удобства
    EMERGENCY_Q_IDX = PATIENT_TYPES["emergency"]["id"]
    CONSULTATION_Q_IDX = PATIENT_TYPES["consultation"]["id"]
    PROCEDURE_Q_IDX = PATIENT_TYPES["procedure"]["id"]

    # Действия
    ACTION_DO_NOTHING = 0
    ACTION_ASSIGN_PATIENT_START = 1 # Действия от 1 до NUM_DOCTORS (включительно)
    ACTION_RESCHEDULE_START = ACTION_ASSIGN_PATIENT_START + NUM_DOCTORS # Действия для перепланирования

    # Общее количество действий
    TOTAL_ACTIONS = ACTION_RESCHEDULE_START + (NUM_DOCTORS * NUM_DOCTORS)

class Patient:
    """
    Представляет пациента в симуляции.
    """
    next_id = 0 # Для уникальных ID пациентов

    def __init__(self, arrival_time_step, patient_type):
        self.id = Patient.next_id
        Patient.next_id += 1
        self.arrival_time_step = arrival_time_step
        self.patient_type = patient_type # Строка: "consultation", "emergency", "procedure"
        self.duration_steps = PolyclinicConstants.PATIENT_TYPES[patient_type]["duration_steps"]
        self.scheduled_doctor = None
        self.scheduled_start_time_step = None
        self.served_time_step = None
        self.waiting_time = 0 # Будет обновляться в среде
        self.is_served = False

    def __repr__(self):
        return f"Patient(ID={self.id}, Type={self.patient_type}, Arrived={self.arrival_time_step}, Scheduled={self.scheduled_start_time_step}, Served={self.is_served})"

# --- Определение среды Gymnasium (скопировано из предыдущего документа) ---
class PolyclinicEnv(gym.Env):
    """
    Среда Gymnasium для динамической оптимизации расписания приёмов в поликлинике.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, arrival_rate=0.5, emergency_ratio=0.1):
        super().__init__()
        self.constants = PolyclinicConstants()

        # Параметры симуляции потока пациентов
        self.arrival_rate = arrival_rate  # Среднее количество прибытий в каждый временной шаг (λ для Пуассона)
        self.emergency_ratio = emergency_ratio  # Доля экстренных случаев

        # Пространство наблюдения
        # [текущее время (1), кол-во экстренных в очереди (1), кол-во консультаций (1), кол-во процедур (1),
        #  свободные слоты для доктора 0 (1), ..., свободные слоты для доктора N-1 (1)]
        low_obs = np.array([
            0, # current_time_step
            0, # emergency_q_count
            0, # consultation_q_count
            0, # procedure_q_count
            *[0 for _ in range(self.constants.NUM_DOCTORS)] # free_slots_per_doctor
        ], dtype=np.int32)
        high_obs = np.array([
            self.constants.MAX_TIME_STEPS_IN_DAY, # current_time_step
            self.constants.MAX_WAITING_ROOM_CAPACITY, # emergency_q_count (upper bound)
            self.constants.MAX_WAITING_ROOM_CAPACITY, # consultation_q_count (upper bound)
            self.constants.MAX_WAITING_ROOM_CAPACITY, # procedure_q_count (upper bound)
            *[self.constants.MAX_TIME_STEPS_IN_DAY for _ in range(self.constants.NUM_DOCTORS)] # free_slots_per_doctor
        ], dtype=np.int32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.int32)

        # Пространство действий
        # 0: Do Nothing
        # 1 to NUM_DOCTORS: Attempt to assign next patient to doctor (action - 1)
        # NUM_DOCTORS + 1 to TOTAL_ACTIONS - 1: Reschedule from_doc_id to to_doc_id
        self.action_space = spaces.Discrete(self.constants.TOTAL_ACTIONS)

        # Инициализация состояния среды
        self.current_time_step = 0
        # self.doctor_schedules: список списков, где schedule[doctor_id][time_step] = patient_id или None.
        # Расширяем расписание за пределы рабочего дня, чтобы учесть переработку
        self.doctor_schedules = [
            [None for _ in range(self.constants.MAX_TIME_STEPS_IN_DAY + self.constants.TOTAL_WORK_MINUTES // self.constants.TIME_STEP_MINUTES)] 
            for _ in range(self.constants.NUM_DOCTORS)
        ]
        self.waiting_patients = collections.deque() # Очередь пациентов, ожидающих приёма
        self.served_patients = [] # Список обслуженных пациентов за день
        self.all_patients_today = [] # Все пациенты, пришедшие за день (для поиска по ID)

        # Метрики для отслеживания
        self.total_waiting_time = 0
        self.total_idle_time_steps = 0
        self.total_overtime_steps = 0
        self.total_served_patients = 0
        self.total_scheduled_patients = 0
        self.daily_metrics = {}

        self.render_mode = render_mode
        self._set_up_render() # Для потенциальной визуализации

    def _set_up_render(self):
        """
        Настройка рендеринга (заглушка, если render_mode не 'human').
        """
        if self.render_mode == 'human':
            print("Render mode is 'human', but no actual rendering implemented yet.")
            pass # Пока не реализуем
        pass

    def _get_obs(self):
        """
        Формирует текущее наблюдение среды.
        """
        emergency_q_count = sum(1 for p in self.waiting_patients if p.patient_type == "emergency")
        consultation_q_count = sum(1 for p in self.waiting_patients if p.patient_type == "consultation")
        procedure_q_count = sum(1 for p in self.waiting_patients if p.patient_type == "procedure")

        doctor_free_slots = []
        for doc_id in range(self.constants.NUM_DOCTORS):
            free_slots = sum(
                1 for i in range(self.current_time_step, self.constants.MAX_TIME_STEPS_IN_DAY)
                if self.doctor_schedules[doc_id][i] is None
            )
            doctor_free_slots.append(free_slots)

        obs = np.array([
            self.current_time_step,
            emergency_q_count,
            consultation_q_count,
            procedure_q_count,
            *doctor_free_slots
        ], dtype=np.int32)
        return obs

    def _get_info(self):
        """
        Формирует дополнительную информацию о состоянии среды для отладки и метрик.
        """
        avg_waiting_time = self.total_waiting_time / self.total_served_patients if self.total_served_patients > 0 else 0

        percent_idle_slots = 0
        if self.current_time_step > 0:
            past_and_current_slots = self.constants.NUM_DOCTORS * self.current_time_step
            past_idle_slots = 0
            for doc_id in range(self.constants.NUM_DOCTORS):
                for t_step in range(self.current_time_step):
                    if t_step < len(self.doctor_schedules[doc_id]) and self.doctor_schedules[doc_id][t_step] is None:
                        past_idle_slots += 1
            percent_idle_slots = (past_idle_slots / past_and_current_slots) * 100 if past_and_current_slots > 0 else 0

        percent_overtime = 0
        total_possible_day_slots = self.constants.NUM_DOCTORS * self.constants.MAX_TIME_STEPS_IN_DAY
        if total_possible_day_slots > 0:
            percent_overtime = (self.total_overtime_steps / total_possible_day_slots) * 100 

        self.daily_metrics = {
            "average_patient_waiting_time": avg_waiting_time,
            "total_served_patients": self.total_served_patients,
            "total_patients_arrived": len(self.all_patients_today),
            "total_waiting_patients": len(self.waiting_patients),
            "total_doctor_idle_steps": self.total_idle_time_steps,
            "total_overtime_steps": self.total_overtime_steps,
            "percent_idle_slots": percent_idle_slots,
            "percent_overtime": percent_overtime,
        }
        return self.daily_metrics

    def reset(self, seed=None, options=None):
        """
        Сбрасывает среду в начальное состояние.
        """
        super().reset(seed=seed)
        Patient.next_id = 0 # Сброс ID пациентов для нового эпизода

        self.current_time_step = 0
        self.doctor_schedules = [
            [None for _ in range(self.constants.MAX_TIME_STEPS_IN_DAY + self.constants.TOTAL_WORK_MINUTES // self.constants.TIME_STEP_MINUTES)]
            for _ in range(self.constants.NUM_DOCTORS)
        ]
        self.waiting_patients = collections.deque()
        self.served_patients = []
        self.all_patients_today = []

        self.total_waiting_time = 0
        self.total_idle_time_steps = 0
        self.total_overtime_steps = 0
        self.total_served_patients = 0
        self.total_scheduled_patients = 0
        self.daily_metrics = {}

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _process_patient_arrivals(self):
        """
        Симулирует прибытие новых пациентов на основе распределения Пуассона.
        """
        num_arrivals = np.random.poisson(self.arrival_rate)
        for _ in range(num_arrivals):
            if len(self.waiting_patients) < self.constants.MAX_WAITING_ROOM_CAPACITY:
                patient_type = "emergency" if np.random.rand() < self.emergency_ratio else np.random.choice(["consultation", "procedure"])
                new_patient = Patient(self.current_time_step, patient_type)
                self.waiting_patients.append(new_patient)
                self.all_patients_today.append(new_patient)

    def _get_patient_by_id(self, patient_id):
        """
        Вспомогательная функция для получения объекта пациента по его ID.
        """
        return next((p for p in self.all_patients_today if p.id == patient_id), None)

    def _clear_patient_schedule(self, patient, doctor_id):
        """
        Очищает слоты, занятые пациентом в расписании указанного врача.
        """
        if patient.scheduled_doctor == doctor_id and patient.scheduled_start_time_step is not None:
            for i in range(patient.duration_steps):
                time_to_clear = patient.scheduled_start_time_step + i
                if 0 <= time_to_clear < len(self.doctor_schedules[doctor_id]):
                    if self.doctor_schedules[doctor_id][time_to_clear] == patient.id:
                        self.doctor_schedules[doctor_id][time_to_clear] = None
    
    def _find_free_slot(self, doctor_id, patient_duration, start_time_step):
        """
        Находит первый свободный слот для пациента указанной продолжительности у врача.
        Возвращает начальный временной шаг слота или -1, если нет свободного.
        """
        for t_step in range(max(self.current_time_step, start_time_step), len(self.doctor_schedules[doctor_id])):
            if t_step + patient_duration <= len(self.doctor_schedules[doctor_id]):
                is_slot_free = True
                for i in range(patient_duration):
                    if self.doctor_schedules[doctor_id][t_step + i] is not None:
                        is_slot_free = False
                        break
                if is_slot_free:
                    return t_step
        return -1

    def _schedule_patient_in_slots(self, patient, doctor_id, start_time_step):
        """
        Назначает пациента в расписание врача на указанные слоты.
        """
        for i in range(patient.duration_steps):
            self.doctor_schedules[doctor_id][start_time_step + i] = patient.id
        
        patient.scheduled_doctor = doctor_id
        patient.scheduled_start_time_step = start_time_step
        self.total_scheduled_patients += 1

    def _reschedule_patient_action(self, from_doctor_id, to_doctor_id):
        """
        Пытается перенести пациента из расписания одного врача в расписание другого.
        Возвращает награду за это действие.
        """
        reward_change = 0

        if from_doctor_id == to_doctor_id:
            return -0.5 # Штраф за некорректное действие

        patient_to_reschedule = None
        
        for t_step in range(self.current_time_step, len(self.doctor_schedules[from_doctor_id])):
            patient_id_at_slot = self.doctor_schedules[from_doctor_id][t_step]
            if patient_id_at_slot is not None:
                current_patient = self._get_patient_by_id(patient_id_at_slot)
                if current_patient and not current_patient.is_served and current_patient.scheduled_start_time_step == t_step:
                    patient_to_reschedule = current_patient
                    break 

        if patient_to_reschedule:
            original_start_time = patient_to_reschedule.scheduled_start_time_step
            original_doctor_id = patient_to_reschedule.scheduled_doctor
            patient_duration = patient_to_reschedule.duration_steps

            self._clear_patient_schedule(patient_to_reschedule, original_doctor_id)

            new_slot_start_time = self._find_free_slot(to_doctor_id, patient_duration, self.current_time_step)

            if new_slot_start_time != -1:
                self._schedule_patient_in_slots(patient_to_reschedule, to_doctor_id, new_slot_start_time)
                
                if new_slot_start_time < original_start_time:
                    reward_change += 5 
                else:
                    reward_change += 2 
            else:
                self._schedule_patient_in_slots(patient_to_reschedule, original_doctor_id, original_start_time) 
                reward_change -= 5 
        else:
            reward_change -= 1 
        
        return reward_change


    def _update_schedules_and_patients(self):
        """
        Обновляет расписания врачей и статусы пациентов.
        """
        for doc_id in range(self.constants.NUM_DOCTORS):
            for t_step in range(self.current_time_step - 1, self.current_time_step): 
                if t_step >= 0 and t_step < len(self.doctor_schedules[doc_id]):
                    patient_id_at_slot = self.doctor_schedules[doc_id][t_step]
                    if patient_id_at_slot is not None:
                        patient = self._get_patient_by_id(patient_id_at_slot)
                        if patient and patient.scheduled_start_time_step is not None and not patient.is_served:
                            scheduled_end_time = patient.scheduled_start_time_step + patient.duration_steps
                            if scheduled_end_time == self.current_time_step:
                                patient.served_time_step = self.current_time_step
                                patient.is_served = True
                                self.served_patients.append(patient)
                                self.total_served_patients += 1
                                patient.waiting_time = patient.served_time_step - patient.arrival_time_step
                                self.total_waiting_time += patient.waiting_time
                                
                                for i in range(patient.duration_steps):
                                    slot_to_clear = patient.scheduled_start_time_step + i
                                    if 0 <= slot_to_clear < len(self.doctor_schedules[doc_id]) and \
                                       self.doctor_schedules[doc_id][slot_to_clear] == patient.id:
                                        self.doctor_schedules[doc_id][slot_to_clear] = None

        for doc_id in range(self.constants.NUM_DOCTORS):
            if self.current_time_step < self.constants.MAX_TIME_STEPS_IN_DAY: 
                if self.doctor_schedules[doc_id][self.current_time_step] is None:
                    is_doctor_idle_at_current_time = True
                    for patient in self.all_patients_today:
                        if patient.scheduled_doctor == doc_id and \
                           patient.scheduled_start_time_step is not None and \
                           not patient.is_served:
                            if patient.scheduled_start_time_step <= self.current_time_step < (patient.scheduled_start_time_step + patient.duration_steps):
                                is_doctor_idle_at_current_time = False
                                break
                    if is_doctor_idle_at_current_time:
                        self.total_idle_time_steps += 1

        for doc_id in range(self.constants.NUM_DOCTORS):
            if self.current_time_step > self.constants.MAX_TIME_STEPS_IN_DAY: 
                if self.current_time_step -1 < len(self.doctor_schedules[doc_id]) and self.doctor_schedules[doc_id][self.current_time_step - 1] is not None:
                    self.total_overtime_steps += 1

    def step(self, action):
        """
        Выполняет один шаг в среде.
        """
        reward = 0
        terminated = False
        truncated = False

        # 1. Применяем действие агента
        if action == self.constants.ACTION_DO_NOTHING:
            pass 

        elif self.constants.ACTION_ASSIGN_PATIENT_START <= action < self.constants.ACTION_RESCHEDULE_START:
            target_doctor_id = action - self.constants.ACTION_ASSIGN_PATIENT_START
            
            patient_to_schedule = None
            
            for p in self.waiting_patients: 
                if p.patient_type == "emergency":
                    patient_to_schedule = p
                    self.waiting_patients.remove(patient_to_schedule) 
                    break
            
            if patient_to_schedule is None:
                for p in self.waiting_patients: 
                    if p.patient_type in ["consultation", "procedure"]:
                        patient_to_schedule = p
                        self.waiting_patients.remove(patient_to_schedule) 
                        break

            if patient_to_schedule:
                found_slot = self._find_free_slot(target_doctor_id, patient_to_schedule.duration_steps, self.current_time_step)

                if found_slot != -1:
                    self._schedule_patient_in_slots(patient_to_schedule, target_doctor_id, found_slot)
                    reward += 10 
                else:
                    self.waiting_patients.appendleft(patient_to_schedule) 
                    reward -= 1 

        elif self.constants.ACTION_RESCHEDULE_START <= action < self.constants.TOTAL_ACTIONS:
            reschedule_action_idx = action - self.constants.ACTION_RESCHEDULE_START
            from_doctor_id = reschedule_action_idx // self.constants.NUM_DOCTORS
            to_doctor_id = reschedule_action_idx % self.constants.NUM_DOCTORS

            reward += self._reschedule_patient_action(from_doctor_id, to_doctor_id)

        # 2. Продвигаем время и обрабатываем прибытия
        self._process_patient_arrivals()
        self.current_time_step += 1
        self._update_schedules_and_patients() 

        # 3. Расчет вознаграждения (штрафы обновляются на каждом шаге)
        waiting_penalty = 0
        for p in self.waiting_patients:
            waiting_penalty += 0.1 
            p.waiting_time += 1 
        reward -= waiting_penalty

        reward -= self.total_idle_time_steps * 0.05 
        reward -= self.total_overtime_steps * 2 

        # 4. Проверка условий завершения эпизода
        if self.current_time_step >= self.constants.MAX_TIME_STEPS_IN_DAY:
            terminated = True 

            self._update_schedules_and_patients() 
            unserved_penalty = 0
            for p in self.waiting_patients:
                unserved_penalty += 5 
            reward -= unserved_penalty

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Отображает текущее состояние среды (заглушка).
        """
        if self.render_mode == 'human':
            pass

    def close(self):
        """
        Очищает ресурсы среды (заглушка).
        """
        if hasattr(self, 'window') and self.window:
            pass

# --- Реализация агента FIFO (Baseline) ---
class FIFOAgent:
    """
    Агент, реализующий стратегию "Первым пришел - первым обслужен" (FIFO)
    с приоритетом для экстренных пациентов.
    """
    def __init__(self, env):
        self.env = env
        self.constants = self.env.constants

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Возвращает действие на основе стратегии FIFO.
        """
        # observation_index_mapping:
        # 0: current_time_step
        # 1: emergency_q_count
        # 2: consultation_q_count
        # 3: procedure_q_count
        # 4 to 4 + NUM_DOCTORS - 1: free_slots_per_doctor

        emergency_q_count = observation[1]
        consultation_q_count = observation[2]
        procedure_q_count = observation[3]
        doctor_free_slots = observation[4:]

        # Приоритет экстренным пациентам
        if emergency_q_count > 0:
            # Находим первого врача с свободным слотом
            for doc_id in range(self.constants.NUM_DOCTORS):
                # Проверяем, есть ли хотя бы один свободный слот у этого доктора
                if doctor_free_slots[doc_id] > 0:
                    # Попытаться назначить экстренного пациента
                    return self.constants.ACTION_ASSIGN_PATIENT_START + doc_id
        
        # Если нет экстренных или некуда их назначить, работаем с обычными
        if consultation_q_count > 0 or procedure_q_count > 0:
            # Находим первого врача с свободным слотом
            for doc_id in range(self.constants.NUM_DOCTORS):
                if doctor_free_slots[doc_id] > 0:
                    # Попытаться назначить обычного пациента
                    return self.constants.ACTION_ASSIGN_PATIENT_START + doc_id
        
        # Если нет пациентов в очереди или все врачи заняты, ничего не делаем
        return self.constants.ACTION_DO_NOTHING

# --- Функция для оценки агентов ---
def evaluate_agent(model, env_fn, num_episodes=100, is_sb3_model=True): # Возвращено к 100 эпизодам
    """
    Оценивает производительность агента на нескольких эпизодах.
    :param model: Агент (модель Stable Baselines3 или FIFOAgent).
    :param env_fn: Функция, которая возвращает новый экземпляр среды.
    :param num_episodes: Количество эпизодов для оценки.
    :param is_sb3_model: True, если модель является моделью Stable Baselines3, иначе False.
    :return: Словарь с усредненными метриками.
    """
    total_rewards = []
    avg_waiting_times = []
    percent_idle_slots = []
    percent_overtimes = []
    total_served_patients = []
    total_patients_arrived = []

    for _ in range(num_episodes):
        env = env_fn()
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            if is_sb3_model:
                action, _ = model.predict(obs, deterministic=True) 
            else:
                action = model.predict(obs) 
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        avg_waiting_times.append(info.get("average_patient_waiting_time", 0))
        percent_idle_slots.append(info.get("percent_idle_slots", 0))
        percent_overtimes.append(info.get("percent_overtime", 0))
        total_served_patients.append(info.get("total_served_patients", 0))
        total_patients_arrived.append(info.get("total_patients_arrived", 0))
        env.close()

    metrics = {
        "avg_total_reward": np.mean(total_rewards),
        "mean_avg_waiting_time": np.mean(avg_waiting_times),
        "mean_percent_idle_slots": np.mean(percent_idle_slots),
        "mean_percent_overtime": np.mean(percent_overtimes),
        "mean_total_served_patients": np.mean(total_served_patients),
        "mean_total_patients_arrived": np.mean(total_patients_arrived),
    }
    return metrics

# --- Функция цели Optuna для PPO ---
def objective(trial: optuna.trial.Trial):
    """
    Функция цели для Optuna: обучает PPO агента и возвращает среднюю награду.
    """
    # Определяем гиперпараметры для оптимизации (полный набор)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = int(trial.suggest_loguniform('n_steps', 256, 2048))
    batch_size = int(trial.suggest_categorical('batch_size', [64, 128, 256, 512]))
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 0.99)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-7, 1e-5)
    vf_coef = trial.suggest_uniform('vf_coef', 0.5, 0.7)

    env_args = {'arrival_rate': 0.7, 'emergency_ratio': 0.2}
    env = make_vec_env(PolyclinicEnv, n_envs=4, env_kwargs=env_args) # Используем 4 параллельные среды

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        device='auto' 
    )

    try:
        # Обучаем модель (большее количество шагов для полноценной оценки в Optuna)
        model.learn(total_timesteps=50_000) # Возвращено к 50_000 шагов для тюнинга

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20) # Возвращено к 20 эпизодам
        env.close()
        return mean_reward
    except Exception as e:
        env.close()
        print(f"Trial failed: {e}")
        return -np.inf 


# --- Основной блок выполнения ---
if __name__ == '__main__':
    # --- 1. Оптимизация гиперпараметров PPO с помощью Optuna ---
    print("--- Запуск оптимизации гиперпараметров PPO с Optuna (полный режим) ---")

    log_dir = "ppo_optuna_logs_full" # Изменено название директории
    os.makedirs(log_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    
    try:
        # Возвращено к большему количеству проб и таймауту
        # n_jobs=1 для простоты запуска, для 2 GPU можно запустить несколько процессов с CUDA_VISIBLE_DEVICES
        study.optimize(objective, n_trials=50, timeout=3600, n_jobs=1) 
    except KeyboardInterrupt:
        print("Optuna optimization interrupted by user.")

    print("\n--- Результаты оптимизации Optuna (полный режим) ---")
    print(f"Лучшая пробная награда: {study.best_value:.2f}")
    print("Лучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # --- 2. Обучение лучшей PPO модели ---
    print("\n--- Обучение лучшей PPO модели (полный режим) ---")
    best_params = study.best_params # Optuna возвращает все оптимизированные параметры
    
    env_args = {'arrival_rate': 0.7, 'emergency_ratio': 0.2}
    train_env = make_vec_env(PolyclinicEnv, n_envs=40, env_kwargs=env_args) # 4 параллельные среды

    best_ppo_model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=best_params['learning_rate'],
        n_steps=int(best_params['n_steps']),
        batch_size=int(best_params['batch_size']),
        gamma=best_params['gamma'],
        gae_lambda=best_params['gae_lambda'],
        ent_coef=best_params['ent_coef'],
        vf_coef=best_params['vf_coef'],
        verbose=1,
        device='auto'
    )

    print("Начало финального обучения PPO (полный режим)...")
    best_ppo_model.learn(total_timesteps=200_000) # Возвращено к 200_000 шагов
    model_path = os.path.join(log_dir, "best_ppo_polyclinic_model_full")
    best_ppo_model.save(model_path)
    print(f"Лучшая PPO модель сохранена по пути: {model_path}")
    train_env.close()

    # --- 3. Инициализация и оценка FIFO агента ---
    print("\n--- Оценка FIFO агента (полный режим) ---")
    fifo_env_creator = lambda: PolyclinicEnv(arrival_rate=0.7, emergency_ratio=0.2)
    fifo_agent = FIFOAgent(fifo_env_creator()) 

    fifo_metrics = evaluate_agent(fifo_agent, fifo_env_creator, num_episodes=100, is_sb3_model=False) # Возвращено к 100 эпизодам
    print("Метрики FIFO агента:")
    for key, value in fifo_metrics.items():
        print(f"  {key}: {value:.4f}")

    # --- 4. Оценка лучшей PPO модели ---
    print("\n--- Оценка лучшей PPO модели (полный режим) ---")
    ppo_eval_env_creator = lambda: PolyclinicEnv(arrival_rate=0.7, emergency_ratio=0.2)
    ppo_metrics = evaluate_agent(best_ppo_model, ppo_eval_env_creator, num_episodes=100, is_sb3_model=True) # Возвращено к 100 эпизодам
    print("Метрики PPO агента (обученного Optuna):")
    for key, value in ppo_metrics.items():
        print(f"  {key}: {value:.4f}")

    # --- 5. Сравнение результатов ---
    print("\n--- Сравнение PPO vs FIFO (полный режим) ---")
    print(f"{'Метрика':<30} | {'PPO':>15} | {'FIFO':>15} | {'Разница (PPO - FIFO)':>25}")
    print("-" * 90)

    comparison_metrics = [
        ("Средняя общая награда", "avg_total_reward"),
        ("Среднее время ожидания пациентов", "mean_avg_waiting_time"),
        ("Средний процент простоя врачей", "mean_percent_idle_slots"),
        ("Средний процент переработки", "mean_percent_overtime"),
        ("Среднее число обслуженных пациентов", "mean_total_served_patients"),
        ("Среднее число прибывших пациентов", "mean_total_patients_arrived")
    ]

    for display_name, key in comparison_metrics:
        ppo_val = ppo_metrics.get(key, np.nan)
        fifo_val = fifo_metrics.get(key, np.nan)
        difference = ppo_val - fifo_val
        print(f"{display_name:<30} | {ppo_val:>15.4f} | {fifo_val:>15.4f} | {difference:>+25.4f}")

    print("\nЗавершено.")
