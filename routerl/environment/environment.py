"""
PettingZoo environment for optimal route choice using SUMO simulator.

"""

import glob
import os

from types import new_class
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from copy import deepcopy as dc
from gymnasium.spaces import Discrete

import functools
import logging
import numpy as np
import pandas as pd
import random
import sys

from routerl.environment import generate_agents
from routerl.environment import SumoSimulator
from routerl.environment import MachineAgent
from routerl.environment import OneActionAgent
from routerl.environment.observations import *
from routerl.keychain import Keychain as kc
from routerl.services import plotter
from routerl.services import Recorder
from routerl.utilities import get_params

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class TrafficEnvironment(AECEnv):
    """
    A PettingZoo AECEnv interface for optimal route choice using SUMO simulator. 
    This environment is designed for the training of human agents (rational decision-makers) 
    and machine agents (reinforcement learning agents).
    
    See `SUMO <https://sumo.dlr.de/docs/>`_ for details on SUMO. \n
    See `PettingZoo <https://pettingzoo.farama.org/>`_ for details on PettingZoo. 
    
    .. note::
        Users can configure the experiment with keyword arguments, see the structure below. 
        Moreover, users can provide custom demand data in ``training_records/agents.csv``.
        You can refer to the structure of such a file `here <https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/main/docs/_static/agents_example.csv>`_.

    Args:
        seed (int, optional): 
            Random seed for reproducibility. Defaults to ``23423``.
        create_agents (bool, optional):
            Whether to create agent data. Defaults to ``True``.
        create_paths (bool, optional):
            Whether to generate paths. Defaults to ``True``.
        action_masks (dict[tuple[int, int], np.ndarray] | None):
            Optional mapping from (origin, destination) pairs to binary action masks.
            Each mask is a 1D NumPy array of 0/1 values with length equal to the action space size. 
            Used only for HumanAgent creation and free flow time retrieval.
        generate_asgn_data (bool):
            Generate additional SUMO_output files (per-timestep departures and snapshots).
        agents (list | None):
            Agents used in the environment. If set to ``None`` the agents will be generated or read from files. Defaults to None. 
        **kwargs (dict, optional): 
            User-defined parameter overrides. These override default values 
            from ``defaults.json`` and allow experiment configuration.
            

    Keyword arguments (see the usage below):
    
        - agent_parameters (dict, optional):
            Agent settings.
            
            - num_agents (int, default=100):
                Total number of agents.
            
            - new_machines_after_mutation (int, default=25):
                Number of humans converted to machines.
            
            - machine_parameters (**dict**):
                Machine agent settings.
                
                - behavior (str, default="selfish"):
                    Route choice behavior.
                    Options: ``selfish``, ``competitive``, ``collaborative``, ``cooperative``, ``social``, ``altruistic``, ``malicious``, ``collectivist``, ``militant``.
                    
                - observed_span (int, default=300):
                    Time window considered for observations.
                    
                - observation_type (str, default="trip_info_eta"):
                    Type of observation.
                    Options: ``previous_agents``, ``previous_agents_plus_start_time``, ``previous_agents_plus_start_time_detector_data``, ``trip_info_eta``.

            - human_parameters (**dict**): 
                Human agent settings.
                
                - model (str, default="gawron"):
                    Decision-making model (options: ``aon``, ``gawron``, ``culo``, ``random``, ``weighted``).
                    
                - beta (float, default=1.5):
                    **Positive value**, multiplier of reward (travel time) used in utility, determines sensitivity.
                    
                - beta_randomness (float, default=0.1):
                    Agent-specific randomness in beta.
                    
                - alpha (float, default=0.2):
                    Human learning rate.

                - deterministic (bool, default=False):
                    Whether ``gawron`` selects the minimum-utility path deterministically instead of sampling stochastically.
                    
                - remember (int, default=5):
                    Number of previous actions to remember for learning, used in ``weighted`` model.

        - environment_parameters (dict, optional):
            Environment settings.
            
            - number_of_days (int, default=1):
                Number of days in the scenario.
                
            - save_every (int, default=1):
                Save the episode data to disk every X days.

        - simulator_parameters (dict, optional): 
            SUMO simulator settings.
            
            - network_name (str, default="csomor"):
                Network name (e.g., ``arterial``, ``cologne``, ``grid``)
                
            - custom_network_folder (str, default="NA"):
                In case of custom network, specify the folder name.
            
            - simulation_timesteps (int, default=3600):
                Total simulation time in seconds.
            
            - sumo_type (str, default="sumo"):
                SUMO execution mode (``sumo`` or ``sumo-gui``).
                
            - stuck_time (int, default=600):
                Number of seconds to tolerate before `teleporting` a stopped vehicle to resolve gridlocks.
                
            - daily_reseed (bool, default=False):
                Whether to change SUMO seed in each reset. If ``False``, the seed will remain constant throughout the simulation.
            
            - use_libsumo (bool, default=False):
                Whether to use libsumo instead of TraCI. Avoid using both ``libsumo=True`` and ``sumo_type=sumo-gui`` at the same time. Visit https://sumo.dlr.de/docs/Libsumo.html for more insight.
            - use_sumo_teleport (bool, default=False):
                If set to ``True`` teleport logic will be handled by SUMO. Otherwise custom python logic will be used.

        - path_generation_parameters (dict, optional):
            Path generation settings.
            
            - number_of_paths (int, default=3):
                Number of routes per OD.
                
            - beta (float, default=-3.0):
                Sensitivity to travel time in path generation.
                
            - weight (str, default="time"):
                Optimization criterion.
                
            - num_samples (int, default=100):
                Number of samples for path generation.

            - path_gen_workers (int, default=4):
                Maximum number of worker processes used for parallel path generation and path visualization.
                
            - origins (str | list[str], default="default"):
                Origin points from the network. (e.g., ``["-25166682#0", "-4936412"]``)
                
            - destinations (str | list[str], default="default"):
                Destination points from the network. (e.g., ``["-115604057#1", "-279952229#4"]``)
                
            - visualize_paths (bool, default=True):
                Whether to visualize generated paths. Visuals will be saved in the ``plotter_parameters/plots_folder``.

        - plotter_parameters (dict, optional): 
            Plotting & logging settings.
            
            - records_folder (str, default="training_records"):
                Directory for training records.
                
            - plots_folder (str, default="plots"):
                Directory for plots.
                
            - plot_choices (str, default="all"):
                Selection of plots to be generated. Options: ``none``, ``basic``, ``all``.
                
            - smooth_by (int, default=50): 
                Smoothing parameter for plots.
                
            - phases (list[int], default=[0, 100]):
                X-axis positions for phase markers.
                
            - phase_names (list[str], default=["Human learning", "Mutation - Machine learning"]):
                Phase names for labeling phase markers.
    
    Usage:
        
        .. rubric:: Case 1
        
        .. code-block:: text
        
            % Your file structure in the beginning
            project_directory/
            |-- your_script.py
            
        .. code-block:: python
        
            >>> # Environment initialization
            ... env = TrafficEnvironment(
            ...     seed=42,
            ...     agent_parameters={
            ...         "num_agents": 5, 
            ...         "new_machines_after_mutation": 1, 
            ...         "machine_parameters": {
            ...             "behavior": "selfish"
            ...             }},
            ...     simulator_parameters={"sumo_type": "sumo-gui"},
            ...     path_generation_parameters={"number_of_paths": 2}
            ... )
            
        .. code-block:: text
        
            % File structure after the initialization:
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv
            |   |-- paths.csv
            |   |-- detector/
            |   |   |--             % to be populated during simulation
            |   |-- episodes/
            |   |   |--             % to be populated during simulation
            |-- plots/
            |   |-- 0_0.png 
            |   |-- ...             % visuals of generated paths for each OD
            |   |-- ...             % to be populated after the experiment
            
        .. raw:: html

            <hr style="border:1px solid #ccc; margin: 20px 0;">
        
        .. rubric:: Case 2
        
        .. code-block:: text
        
            % Your file structure in the beginning
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv      % your custom demand, conforming to the structure
        
        .. warning::
            Demand data in ``agents.csv`` should be aligned with the specified 
            experiment settings (e.g., number of agents, number of origins and destinations, etc.).
            
        .. code-block:: python
        
            >>> env = TrafficEnvironment(
            ...     create_agents=False, # Environment will use your agent data
            ...     agent_parameters={
            ...         "new_machines_after_mutation": 10, 
            ...         "machine_parameters": {
            ...             "behavior": "selfish"
            ...             }},
            ...     simulator_parameters={"network_name": "arterial"},
            ...     path_generation_parameters={"number_of_paths": 3}
            ... )
            
        .. code-block:: text
        
            % File structure after the initialization:
            project_directory/
            |-- your_script.py
            |-- training_records/
            |   |-- agents.csv      % stays the same, used for agent generation
            |   |-- paths.csv
            |   |-- detector/
            |   |   |--             % to be populated during simulation
            |   |-- episodes/
            |   |   |--             % to be populated during simulation
            |-- plots/
            |   |-- 0_0.png 
            |   |-- ...             % visuals of generated paths for each OD
            |   |-- ...             % to be populated after the experiment
            
        .. warning::
            Same approach does not translate to path generation.\n
            ``paths.csv`` is mainly used for visualization purposes. 
            For SUMO to operate correctly, a ``route.rou.xml`` 
            should be generated inside the ``routerl/networks/<net_name>/`` folder.\n
            It is advised to generate paths in each experiment providing a random seed,
            or set ``create_paths=False`` only when above criteria is met.

    Attributes:
        day (int): Current day index in the simulation.
        human_learning (bool): Whether human agents are learning.
        number_of_days (int): Number of days to simulate.
        action_space_size (int): Size of the action space.
        recorder (Recorder): Object for recording simulation data.
        simulator (SumoSimulator): SUMO simulator instance.
        all_agents (list): List of all agent objects.
        machine_agents (list): List of all machine agent objects.
        human_agents (list): List of all human agent objects.
        last_episode_had_teleports (bool): Whether any agents were teleported in the last episode.
        last_episode_travel_times (list): List of machine agents' travel times in the last episode.
    """
    
    metadata = {
        "render_modes": ["human"],
        "name": "TrafficEnvironment",
    }

    def __init__(self,
                 seed: int = 23423,
                 create_agents: bool = True,
                 create_paths: bool = True,
                 save_detectors_info: bool = False,
                 action_masks: dict = None,
                 generate_asgn_data: bool = False,
                 agents: list = None,
                 **kwargs) -> None:

        super().__init__()
        self.kwargs = kwargs
        self.render_mode = None

        # Read default parameters, update with kwargs
        defaults_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), kc.DEFAULTS_FILE)
        params = get_params(defaults_path, resolve=True, update=kwargs)

        self.environment_params = params[kc.ENVIRONMENT]
        self.simulation_params = params[kc.SIMULATOR]
        self.agent_params = params[kc.AGENTS]
        self.plotter_params = params[kc.PLOTTER]
        self.path_gen_params = params[kc.PATH_GEN] if create_paths else None

        self.travel_times_list = []
        self.day = 0
        self.human_learning = True
        self.machine_same_start_time = []
        self.actions_timestep = []
        self.save_detectors_info = save_detectors_info
        self.last_episode_had_teleports = False
        self.last_episode_travel_times = list(self.travel_times_list)

        self.number_of_days = self.environment_params[kc.NUMBER_OF_DAYS]
        self.save_every = self.environment_params[kc.SAVE_EVERY]
        self.action_space_size = self.environment_params[kc.ACTION_SPACE_SIZE]
        self._set_seed(seed)

        self.action_masks = action_masks
        self.use_action_masks = self.action_masks is not None # for the environment
        self.use_clustered_routes = self.action_masks is not None # for the simulator

        self.recorder = Recorder(self.plotter_params)
        self.simulator = SumoSimulator(self.simulation_params, self.path_gen_params, seed, not create_agents, save_detectors_info, generate_asgn_data, self.use_clustered_routes)

        self.all_agents = generate_agents(self.agent_params, self.get_free_flow_times(), create_agents, seed, self.action_masks) if agents == None else agents
        self.machine_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_MACHINE]
        self.human_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_HUMAN]
        self.possible_agents = list()

        if len(self.machine_agents):
            self._initialize_machine_agents()
        if not self.human_agents:
            self.human_learning = False
        logging.info(f"There are {len(self.human_agents)} human and {len(self.machine_agents)} machine agents.")

        self.episode_actions = dict()
        self.episode_observations = dict()
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_futures = []

    def __str__(self):
        message = f"TrafficEnvironment with {len(self.all_agents)} agents.\
            \n{len(self.machine_agents)} machines and {len(self.human_agents)} humans.\
            \nMachines: {sorted(self.machine_agents, key=lambda agent: agent.id)}\
            \nHumans: {sorted(self.human_agents, key=lambda agent: agent.id)}"
        return message

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        logging.info(f"Seed set to {seed}.")

    def _initialize_machine_agents(self) -> None:

        ## Sort machine agents based on their start_time
        sorted_machine_agents = sorted(self.machine_agents, key=lambda agent: agent.start_time)
        self.possible_agents = [str(agent.id) for agent in sorted_machine_agents]
        self.n_agents = len(self.possible_agents)

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        ## Initialize the observation object
        self.observation_obj = self.get_observation_function()
        self._observation_spaces = self.observation_obj.observation_space()

        self._action_spaces = {
            agent: Discrete(self.simulation_params[kc.NUMBER_OF_PATHS]) for agent in self.possible_agents
        }

        logging.info("\nMachine's observation space is: %s ", self._observation_spaces)
        logging.info("Machine's action space is: %s", self._action_spaces)

    ################################
    ######## Control methods #######
    ################################

    def start(self) -> None:
        """Start the connection with SUMO.
        
        Returns:
            None
        """

        self.simulator.start()

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Resets the environment.
        
        Args:
            seed (int, optional): Seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.
            
        Returns:
            observations (dict): observations.
            infos (dict): dictionary of information for the agents.
        """
        self.episode_actions = dict()
        self.travel_times_list = list()
        self.actions_timestep = list()
        self.machine_same_start_time = list()
        self.episode_observations = dict()
        self.last_episode_had_teleports = False
        self.simulator.reset()
        self.agents = copy(self.possible_agents)
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.rewards_humans = {agent.id: 0 for agent in self.human_agents}

        if len(self.machine_agents) > 0:
            self._agent_selector = agent_selector(self.possible_agents)
            self.agent_selection = self._agent_selector.next()
            self.observations = self.observation_obj.reset_observation()
        else:
            self.observations = {}

        infos = {a: {} for a in self.possible_agents}

        return self.observations, infos

    def step(self, machine_action: int = None) -> None:
        """Step method.

        Takes an action for the current agent (specified by `agent_selection`) and updates
        various parameters including rewards, cumulative rewards, terminations, truncations,
        infos, and agent_selection. Also updates any internal state used by `observe()`.

        Args:
            machine_action (int, optional):
                The action to be taken by the machine agent. Defaults to None.
            
        Returns:
            None
        """

        # If there are machines in the system
        if self.possible_agents:
            if (self.terminations[self.agent_selection]
                    or self.truncations[self.agent_selection]):
                # handles stepping an agent which is already dead
                # accepts a None action for the one agent, and moves the agent_selection to
                # the next dead agent,  or if there are no more dead agents, to the next live agent
                self._was_dead_step(machine_action)
                return

            agent = self.agent_selection

            # The cumulative reward of the last agent must be 0
            self._cumulative_rewards[agent] = 0
            self.simulation_loop(machine_action, agent)

            # Collect rewards if it is the last agent to act
            if self._agent_selector.is_last():
                # Increase day number
                self.day += 1
                
                # Calculate marginal cost
                #marginal_cost = self.calculate_marginal_cost()
                #self.recorder.remember_marginal_costs(marginal_cost, self.day-1) #TODO awful improve

                # Calculate approximated marginal cost
                approximated_marginal_cost = self.calculate_approximated_marginal_cost()

                # Calculate the rewards
                self._assign_rewards(approximated_marginal_cost)

                # The episode ends when we complete episode_length days
                self.truncations = {agent: not (self.day % self.number_of_days) for agent in self.agents}
                self.terminations = {agent: not (self.day % self.number_of_days) for agent in self.agents}
                self.infos = {agent: {} for agent in self.agents}
                self.observations = self.observation_obj(self.all_agents)
                self._reset_episode()
            else:
                # no rewards are allocated until all players give an action
                self._clear_rewards()
                self.agent_selection = self._agent_selector.next()

            # Adds .rewards to ._cumulative_rewards
            self._accumulate_rewards()

        # If there are only humans in the system
        else:
            self.simulation_loop(machine_action=0, machine_id=0)
            self.day = self.day + 1
            self._assign_rewards()
            self._reset_episode()

    def close(self) -> None:
        """Not implemented.

        Returns:
            None
        """
        pass
    
    def stop_simulation(self) -> None:
        """End the simulation.

        Returns:
            None
        """

        self.simulator.stop()
        for future in self.pending_futures:
            future.result()
        self.executor.shutdown(wait=True)

    def observe(self, agent: str) -> np.ndarray:
        """Retrieve the observations for a specific agent.

        Args:
            agent (str): The identifier for the agent whose observations are to be retrieved.
            
        Returns:
            self.observation_obj.agent_observations(agent) (np.ndarray): The observations for the specified agent.
        """
        for machine in self.machine_agents:
            if str(machine.id) == agent:
                break

        # If the agent's turn hasn't come and the start time is bigger than the simulator timestep return an "empty observation"
        # The agent hasn't acted yet so only the start time is meaningful
        if agent != self.agent_selection and machine.start_time > self.simulator.timestep:
            observation = self.observation_obj.observations[agent].copy()
            return observation
        
        return self.observation_obj.agent_observations(agent, self.all_agents, self.agent_selection, self.travel_times_list)

    #########################
    ### Mutation function ###
    #########################

    def mutation(self, disable_human_learning: bool = True, mutation_start_percentile: int = 25) -> None:
        """Perform mutation by converting selected human agents into machine agents.

        This method identifies a subset of human agents that start after the 25th percentile of start times of
        other vehicles, removes a specified number of these agents, and replaces them with machine agents.

        Args:
            disable_human_learning (bool, default=True): Boolean flag to disable human agents.
            mutation_start_percentile (int, default=25): The percentile threshold for selecting human agents for mutation. Set to -1 to disable this filter.
            
        Returns:
            None
            
        Raises:
            ValueError: If there are insufficient human agents available for mutation.
        """

        logging.info("Mutation is about to happen!\n")
        logging.info("There were %s human agents.\n", len(self.human_agents))

        if mutation_start_percentile == -1:
            filtered_human_agents = self.human_agents.copy()
        else:
            start_times = [human.start_time for human in self.human_agents]
            percentile = np.percentile(start_times, mutation_start_percentile)
            filtered_human_agents = [human for human in self.human_agents if human.start_time > percentile]

        number_of_machines_to_be_added = self.agent_params[kc.NEW_MACHINES_AFTER_MUTATION]

        if len(filtered_human_agents) < number_of_machines_to_be_added:
            raise ValueError(
                f"Insufficient human agents for mutation. Required: {number_of_machines_to_be_added}, "
                f"Available: {len(filtered_human_agents)}.\n"
                f"Decrease the number of machines to be added after the mutation.\n"
            )

        for _ in range(0, number_of_machines_to_be_added):
            random_human = random.choice(filtered_human_agents)

            self.human_agents.remove(random_human)
            filtered_human_agents.remove(random_human)

            self.machine_agents.append(MachineAgent(random_human.id,
                                                    random_human.start_time,
                                                    random_human.origin,
                                                    random_human.destination,
                                                    self.agent_params[kc.MACHINE_PARAMETERS],
                                                    self.action_space_size))
            self.possible_agents.append(str(random_human.id))

        self.n_agents = len(self.possible_agents)
        self.all_agents = self.machine_agents + self.human_agents

        if disable_human_learning:  self.human_learning = False

        logging.info(f"Now there are {len(self.human_agents)} human agents.")

        self._initialize_machine_agents()

    #########################
    ##### Help functions ####
    #########################

    def get_observation(self) -> tuple:
        """Retrieve the current observation from the simulator.

        This method returns the current timestep of the simulation and the values of the episode actions.

        Returns:
            tuple: A tuple containing the current timestep and the episode actions.
        """

        return self.simulator.timestep, self.episode_actions.values()

    def _help_step(self, actions: list[tuple]) -> dict:

        for agent, action in actions:
            observation = kc.NOT_AVAILABLE
            if agent.kind == kc.TYPE_MACHINE:
                observation = self.episode_observations.get(agent.id, kc.NOT_AVAILABLE)
            action_dict = {kc.AGENT_ID: agent.id,
                           kc.AGENT_KIND: agent.kind,
                           kc.ACTION: action,
                           kc.AGENT_ORIGIN: agent.origin,
                           kc.AGENT_DESTINATION: agent.destination,
                           kc.AGENT_START_TIME: agent.start_time,
                           kc.AGENT_OBSERVATION: observation}
            self.simulator.add_vehicle(action_dict)
            self.episode_actions[agent.id] = action_dict
        timestep, stopped_vehicles_info, arrivals, teleported = self.simulator.step()

        if self.save_detectors_info == True:
            self._save_detectors_info(stopped_vehicles_info)

        travel_times = dict()
        for veh_id in arrivals:
            if veh_id not in teleported:
                agent_id = int(veh_id)
                travel_times[agent_id] = ({kc.TRAVEL_TIME:
                                            (timestep - self.episode_actions[agent_id][kc.AGENT_START_TIME]) / 60.0})
                travel_times[agent_id].update(self.episode_actions[agent_id])
            
        for veh_id in teleported:
            agent_id = int(veh_id)
            travel_times[agent_id] = ({kc.TRAVEL_TIME: self.simulator.simulation_length / 60.0})
            travel_times[agent_id].update(self.episode_actions[agent_id])

        if teleported:
            self.last_episode_had_teleports = True

        return travel_times.values()
    
    def _save_detectors_info(self, stopped_vehicles_info):
        folder = self.plotter_params[kc.RECORDS_FOLDER] + '/' + kc.DETECTOR_STOPPED_VEHICLES
        os.makedirs(folder, exist_ok=True)

        if (self.simulator.timestep == 1):
             [os.remove(f) for f in glob.glob(f"{folder}/*.csv")]

        csv_file_path = f"{folder}/stopped_vehicles{self.simulator.timestep - 1}.csv"

        df = pd.DataFrame(stopped_vehicles_info, columns=["time", "detector", "vehicle_id"])
        df.to_csv(csv_file_path, index=False)

    def _reset_episode(self) -> None:

        # Snapshot travel_times_list before clearing
        self.last_episode_travel_times = list(self.travel_times_list)

        detectors_dict = self.simulator.reset()

        if self.possible_agents:
            self._agent_selector = agent_selector(self.possible_agents)
            self.agent_selection = self._agent_selector.next()

        if self.day % self.save_every == 0 and self.plotter_params.get(kc.RECORD, True):
            dc_episode, dc_ep_observations, dc_agents, dc_detectors = dc(self.day), dc(self.travel_times_list), dc(self.all_agents), dc(detectors_dict)
            recording_task = self.executor.submit(self._record, dc_episode, dc_ep_observations, dc_agents, dc_detectors)
            self.pending_futures.append(recording_task)
        
        # Reset observations
        if len(self.machine_agents) > 0:
            self.observations = self.observation_obj.reset_observation()

        self.travel_times_list = []
        self.episode_actions = dict()
        self.episode_observations = dict()

    def _assign_rewards(self, marginal_cost = None) -> None:

        for agent in self.all_agents:
            if agent.kind == 'Human':
                reward = agent.get_reward(self.travel_times_list)
            else:
                reward = agent.get_reward(
                    self.travel_times_list, 
                    group_vicinity=self.agent_params[kc.MACHINE_PARAMETERS][kc.GROUP_VICINITY], 
                    marginal_cost_matrix=marginal_cost
                )

            # Add the reward in the travel_times_list
            for agent_entry in self.travel_times_list:
                if agent.id == agent_entry[kc.AGENT_ID]:
                    self.travel_times_list.remove(agent_entry)
                    agent_entry[kc.REWARD] = reward
                    self.travel_times_list.append(agent_entry)

            # Save machine's rewards based on PettingZoo standards
            if agent.kind == 'AV':
                self.rewards[str(agent.id)] = reward

            # Human learning
            elif self.human_learning:
                agent.learn(agent.last_action, self.travel_times_list)

    ###########################
    ##### Simulation loop #####
    ###########################

    def simulation_loop(self, machine_action: int, machine_id: int) -> None:
        """This function contains the integration of the agent's actions to SUMO.

        We iterate through all the time steps of the simulation.
        For each timestep there are none, one or more than one agents type (humans, machines) that start.
        If more than one machine agents have the same start time, we break from this function because
        we need to take the agent's action from the STEP function.

        Args:
            machine_action (int): The id of the machine agent whose action is to be performed.
            machine_id (int): The id of the machine agent whose action is to be performed.
            
        Returns:
            None
        """

        agent_action = False
        while (
                self.simulator.timestep < self.simulation_params[kc.SIMULATION_TIMESTEPS]
                or len(self.travel_times_list) < len(self.all_agents)
        ):

            # If there are more than one machines with the same start time
            # the humans should act once
            if not self.actions_timestep:
                for human in self.human_agents:
                    if human.start_time == self.simulator.timestep:
                        action = human.act(0)
                        human.last_action = action
                        self.actions_timestep.append((human, action))

            for machine in self.machine_agents:
                if machine.start_time == self.simulator.timestep:

                    # In case there are machine agents that have the same start time, but it's not their turn
                    if str(machine.id) != machine_id:

                        # If some machines have the same start time, and they haven't acted yet
                        if (
                                (machine not in self.machine_same_start_time)
                                and not any(machine == item[0] for item in self.actions_timestep)
                        ):
                            self.machine_same_start_time.append(machine)
                        continue
                    else:
                        # Machine acting
                        observation = self.observe(str(machine.id))
                        self.episode_observations[machine.id] = self._serialize_observation(observation)
                        machine.last_action = machine_action
                        self.actions_timestep.append((machine, machine_action))
                        #self.actions_timestep.append((machine, machine_action if machine_action is not None else machine.default_action)) #TODO inspect

                        # The machine acted should be deleted from the self.machine_same_start_time list
                        if machine in self.machine_same_start_time:
                            self.machine_same_start_time.remove(machine)

                        # If the machine isn't the last agent to act then we need to step again for the next agent
                        if not self._agent_selector.is_last():
                            agent_action = True

            # If all machines that have start time as the simulator timestep acted
            if not self.machine_same_start_time:
                travel_times = self._help_step(self.actions_timestep)

                for agent_dict in travel_times:
                    self.travel_times_list.append(agent_dict)

                self.actions_timestep = []
                self.machine_same_start_time = []

            # If the machine agent that had turn acted
            if agent_action:
                agent_action = False
                break

    def _serialize_observation(self, observation: np.ndarray) -> str:
        if isinstance(observation, np.ndarray):
            observation = observation.tolist()
        elif isinstance(observation, tuple):
            observation = list(observation)
        if isinstance(observation, list):
            return ",".join(map(str, observation))
        return str(observation)

    ###########################
    ##### Free flow times #####
    ###########################

    def get_free_flow_times(self) -> dict:
        """Retrieve free flow times for all origin-destination pairs from the simulator paths data.

        Returns:
            ff_dict (dict): A dictionary where keys are tuples of origin and destination,
                            and values are lists of free flow times.
        """

        paths_df = pd.read_csv(self.simulator.paths_csv_file_path)

        if not self.use_action_masks:
            origins = paths_df[kc.ORIGINS].unique()
            destinations = paths_df[kc.DESTINATIONS].unique()
            ff_dict = {(o, d): list() for o in origins for d in destinations}

            for _, row in paths_df.iterrows():
                ff_dict[(row[kc.ORIGINS], row[kc.DESTINATIONS])].append(row[kc.FREE_FLOW_TIME])
        else:
            # Pad invalid actions (missing paths) with large values
            num_paths = self.agent_params[kc.ACTION_SPACE_SIZE]

            cluster_ff_dict = {}
            for _, row in paths_df.iterrows():
                key = (int(row[kc.ORIGINS]), int(row[kc.DESTINATIONS]))
                if key not in cluster_ff_dict:
                    cluster_ff_dict[key] = {} # dict with cluster: fft mapping
                cluster = int(row["cluster"]) # add to kc?
                cluster_ff_dict[key][cluster] = float(row[kc.FREE_FLOW_TIME])

            ff_dict = {}
            for key, cluster_ff in cluster_ff_dict.items():
                ff_dict[key] = [cluster_ff.get(i, 1e9) for i in range(num_paths)]

        return ff_dict

    ############################
    ##### Disc operations ######
    ############################

    def _record(self, episode: int, ep_observations: dict, agents: list, detectors_dict: dict) -> None:
        zero_space = [0] * self.action_space_size
        cost_tables = [
            {
                kc.AGENT_ID: agent.id,
                kc.COST_TABLE: getattr(agent.model, 'cost', zero_space) if hasattr(agent, 'model') else zero_space
            }
            for agent in agents
        ]
        self.recorder.record(episode, ep_observations, cost_tables, detectors_dict)

    def plot_results(self) -> None:
        """Method that plot the results of the simulation.

        Returns:
            None
        """

        plotter(self.plotter_params)

    ############################
    ### PettingZoo functions ###
    ############################

    def render(self) -> None:
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Method that returns the observation space of the agent.

        Args:
            agent (str): The agent name.
        Returns:
            self._observation_spaces[agent] (Any): The observation space of the agent.
        """

        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Method that returns the action space of the agent.

        Args:
            agent (str): The agent name.
        Returns:
            self._action_spaces[agent] (Any): The action space of the agent.
        """

        return self._action_spaces[agent]

    #####################################################
    ### Decide on the observation function to be used ###
    #####################################################

    def get_observation_function(self) -> Observations:
        """Returns an observation object based on the provided parameters.

        Returns:
            Observations: An observation object.
        Raises:
            ValueError: If model is unknown.
        """

        params = self.agent_params[kc.MACHINE_PARAMETERS]
        observation_type = params[kc.OBSERVATION_TYPE]
        if observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME:
            return PreviousAgentStartPlusStartTime(self.machine_agents,
                                                   self.human_agents,
                                                   self.simulation_params,
                                                   self.agent_params)
        elif observation_type == kc.PREVIOUS_AGENTS:
            return PreviousAgentStart(self.machine_agents,
                                      self.human_agents,
                                      self.simulation_params,
                                      self.agent_params)
        elif observation_type == kc.PREVIOUS_AGENTS_PLUS_START_TIME_DETECTOR_DATA:
            if self.save_detectors_info == False:
                raise Exception("Detector info saving is disabled. Please set 'self.save_detectors_info = True' to proceed or change the observation type.")
            
            return PreviousAgentStartPlusStartTimeDetectorData(self.machine_agents,
                                      self.human_agents,
                                      self.simulation_params,
                                      self.plotter_params,
                                      self.agent_params,
                                      self.simulator)
        elif observation_type == kc.TRIP_INFO_ETA:
            return TripInfoWithETA(self.machine_agents,
                                 self.human_agents,
                                 self.simulation_params,
                                 self.agent_params,
                                 self.get_free_flow_times())
        else:
            raise ValueError('[MODEL INVALID] Unrecognized observation type: ' + observation_type)

    # marginal_cost[agent_i][agent_j] is the cost that agent_i imposed on agent_j
    def calculate_marginal_cost(self, machines_to_all: bool = True):
        executor = ProcessPoolExecutor(
            max_workers=2, 
            mp_context=None, 
            initializer=_MarginalCostWorker.initWorker, 
            initargs=(
                cp.deepcopy(self.all_agents), 
                cp.deepcopy(self.travel_times_list), 
                cp.deepcopy(self.seed), 
                cp.deepcopy(machines_to_all),
                cp.deepcopy(self.kwargs)
            ), 
        )

        #machine_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_MACHINE]
        agent_to_calculate_ids = [agent.id for agent in self.machine_agents]
        #print(agent_to_calculate_ids)
        result = executor.map(_MarginalCostWorker.task, agent_to_calculate_ids) 

        return dict(zip(agent_to_calculate_ids, result))
        #return marginal_cost_calculation

    def get_previous_actions(
        self,
        agents,
        av_agent_id,
    ):
        av_agent = next(
            (
                agent
                for agent in agents
                if agent["id"] == av_agent_id
            ),
            None,
        )

        if av_agent is None:
            raise ValueError(
                f"Agent {av_agent_id} was not found."
            )

        if av_agent.get("kind") != "AV":
            raise ValueError(
                f"Agent {av_agent_id} is not an AV agent."
            )

        relevant_agents = sorted(
            [
                agent
                for agent in agents
                if (
                    agent["start_time"]
                    < av_agent["start_time"]
                )
            ]
            + [av_agent],
            key=lambda agent: (
                agent["start_time"],
                agent["id"],
            ),
        )

        return {
            "agent_id": av_agent["id"],
            "start_time": av_agent["start_time"],
            "agent_ids": [
                agent["id"]
                for agent in relevant_agents
            ],
            "actions": [
                agent["action"]
                for agent in relevant_agents
            ],
            "travel_times": [
                agent["travel_time"]
                for agent in relevant_agents
            ],
        }
    
    def assign_joint_action_to_cluster(
        self,
        result,
        cluster_csv=(
            "agent_actions_with_cluster.csv"
        ),
        theta=2,
        acceptance_quantile=0.95,
    ):
        """
        Assign a partial joint action to the nearest cluster using
        the same distance objective as the clustering.

        result must contain:
            result["agent_ids"]
            result["travel_times"]

        Returns
        -------
        dict containing:
            best_cluster
            belongs_to_cluster
            distance
            acceptance_threshold
            ranking
        """

        df = pd.read_csv(cluster_csv)

        required_columns = {
            "simulation_id",
            "agent_id",
            "cluster",
            "travel_time",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        if len(result["agent_ids"]) != len(
            result["travel_times"]
        ):
            raise ValueError(
                "agent_ids and travel_times must have "
                "the same length."
            )

        observed = pd.DataFrame(
            {
                "agent_id": result["agent_ids"],
                "observed_time": result["travel_times"],
            }
        )


        # Agent-specific centroid for every cluster.
        centroids = (
            df.groupby(
                ["cluster", "agent_id"],
                as_index=False,
            )["travel_time"]
            .mean()
            .rename(
                columns={
                    "travel_time": "centroid_time"
                }
            )
        )

        # Add centroid values to every training observation.
        training = df.merge(
            centroids,
            on=["cluster", "agent_id"],
            how="left",
            validate="many_to_one",
        )

        rows = []

        for cluster_id, cluster_centroid in (
            centroids.groupby("cluster")
        ):
            matched = observed.merge(
                cluster_centroid[
                    ["agent_id", "centroid_time"]
                ],
                on="agent_id",
                how="inner",
            )

            coverage = (
                len(matched) / len(observed)
            )

            if matched.empty:
                continue

            difference = (
                matched["observed_time"]
                - matched["centroid_time"]
            )

            individual_powered_differences = (
                np.abs(difference) ** theta
            )

            # Sum over individual agents, matching the sum over i
            # in the mathematical objective.
            objective = individual_powered_differences.sum()

            distance = objective ** (1 / theta)

            # Use exactly the agents that were matched for this cluster.
            matched_agent_ids = matched["agent_id"].unique()

            cluster_training = training[
                (training["cluster"] == cluster_id)
                & (
                    training["agent_id"].isin(
                        matched_agent_ids
                    )
                )
            ].copy()

            cluster_training["powered_error"] = (
                np.abs(
                    cluster_training["travel_time"]
                    - cluster_training["centroid_time"]
                )
                ** theta
            )

            # Calculate one summed objective and one agent count
            # for each historical simulation.
            training_summary = (
                cluster_training
                .groupby("simulation_id")
                .agg(
                    training_objective=(
                        "powered_error",
                        "sum",
                    ),
                    number_of_agents=(
                        "agent_id",
                        "nunique",
                    ),
                )
            )

            # Keep only historical simulations that contain all
            # agents used in the current cluster comparison.
            training_summary = training_summary[
                training_summary["number_of_agents"]
                == len(matched_agent_ids)
            ]

            training_distances = (
                training_summary["training_objective"]
                ** (1 / theta)
            ).dropna()

            if training_distances.empty:
                continue

            threshold = training_distances.quantile(
                acceptance_quantile
            )

            # Empirical cluster compatibility.
            #
            # This is the proportion of historical cluster members
            # that were at least as far from the centroid as the
            # current joint action.
            compatibility = (
                1
                + (training_distances >= distance).sum()
            ) / (
                len(training_distances) + 1
            )

            rows.append(
                {
                    "cluster": cluster_id,
                    "objective": float(objective),
                    "distance": float(distance),
                    "threshold": float(threshold),
                    "compatibility": float(
                        compatibility
                    ),
                    "matched_agents": len(matched),
                    "total_agents": len(observed),
                    "coverage": float(coverage),
                    "training_simulations": len(
                        training_distances
                    ),
                }
            )

        ranking = pd.DataFrame(rows)

        if ranking.empty:
            raise ValueError(
                "No agents from the result were found "
                "in the cluster CSV."
            )

        ranking = ranking.sort_values(
            [
                "objective",
                "cluster",
            ]
        ).reset_index(drop=True)

        ranking["rank"] = (
            np.arange(len(ranking)) + 1
        )

        # ----------------------------------------------------
        # Identify whose partial joint-action information
        # is being evaluated.
        # ----------------------------------------------------
        query_agent_id = result.get("agent_id")

        # For acceptance_quantile = 0.95, alpha = 0.05.
        alpha = 1 - acceptance_quantile

        ranking["query_agent_id"] = query_agent_id

        # Decide compatibility using the empirical
        # compatibility score.
        ranking["compatible"] = (
            ranking["compatibility"] > alpha
        )

        # Keep the distance-threshold decision as a
        # separate diagnostic.
        ranking["within_threshold"] = (
            ranking["distance"]
            <= ranking["threshold"]
        )

        best = ranking.iloc[0]

        compatible_clusters = (
            ranking.loc[
                ranking["compatible"],
                "cluster",
            ]
            .tolist()
        )

        return {
            "query_agent_id": query_agent_id,
            "number_of_known_agents": len(
                result["agent_ids"]
            ),
            "best_cluster": best["cluster"],
            "belongs_to_cluster": bool(
                best["compatible"]
            ),
            "best_cluster_compatibility": float(
                best["compatibility"]
            ),
            "compatible_clusters": compatible_clusters,
            "distance": float(
                best["distance"]
            ),
            "acceptance_threshold": float(
                best["threshold"]
            ),
            "coverage": float(
                best["coverage"]
            ),
            "ranking": ranking,
        }

     # marginal_cost[agent_i][agent_j] is the cost that agent_i imposed on agent_j
    def calculate_marginal_cost_for_agent_id(self, agent_to_calculate_ids, travel_times_list, machines_to_all: bool = True):
            executor = ProcessPoolExecutor(
                max_workers=1, 
                mp_context=None, 
                initializer=_MarginalCostWorker.initWorker, 
                initargs=(
                    cp.deepcopy(self.all_agents), 
                    cp.deepcopy(travel_times_list), 
                    cp.deepcopy(self.seed), 
                    cp.deepcopy(machines_to_all),
                    cp.deepcopy(self.kwargs)
                ), 
            )
    
            #machine_agents = [agent for agent in self.all_agents if agent.kind == kc.TYPE_MACHINE]
            #agent_to_calculate_ids = [agent.id for agent in self.machine_agents]
            #print(agent_to_calculate_ids)
            result = executor.map(_MarginalCostWorker.task, agent_to_calculate_ids) 
    
            return dict(zip(agent_to_calculate_ids, result))
            #return marginal_cost_calculation

    def _compute_cluster_total_marginal_cost(
        self,
        machine_id,
        cluster_id,
        known_agent_ids,
        cluster_data,
        current_records_by_agent,
        theta,
    ):
        """
        Compute the marginal cost of one AV in one cluster.

        This is the expensive part. It should only be called when
        the (machine_id, cluster_id) combination is not cached.

        Returns
        -------
        dict | None
            Cache entry containing the per-agent differences and
            their summed external marginal cost.
        """

        machine_id = int(machine_id)
        cluster_id = int(cluster_id)

        selected_cluster_data = cluster_data.loc[
            (
                cluster_data["cluster"] == cluster_id
            )
            & (
                cluster_data["agent_id"].isin(
                    known_agent_ids
                )
            )
        ].copy()

        if selected_cluster_data.empty:
            print(
                f"Cluster {cluster_id} has no records for "
                f"machine {machine_id}'s known agents."
            )
            return None

        # ========================================================
        # Compute the agent-specific travel-time centroid
        # ========================================================
        cluster_centroid = (
            selected_cluster_data
            .groupby(
                "agent_id",
                as_index=False,
            )["travel_time"]
            .mean()
            .rename(
                columns={
                    "travel_time": "centroid_time"
                }
            )
        )

        if cluster_centroid.empty:
            return None

        centroid_candidates = (
            selected_cluster_data.merge(
                cluster_centroid,
                on="agent_id",
                how="inner",
                validate="many_to_one",
            )
        )

        centroid_candidates["powered_error"] = (
            np.abs(
                centroid_candidates["travel_time"]
                - centroid_candidates[
                    "centroid_time"
                ]
            )
            ** theta
        )

        number_of_centroid_agents = int(
            cluster_centroid[
                "agent_id"
            ].nunique()
        )

        # ========================================================
        # Find the historical simulation closest to the centroid
        # ========================================================
        simulation_errors = (
            centroid_candidates
            .groupby(
                "simulation_id",
                as_index=False,
            )
            .agg(
                centroid_objective=(
                    "powered_error",
                    "sum",
                ),
                number_of_agents=(
                    "agent_id",
                    "nunique",
                ),
            )
        )

        simulation_errors = (
            simulation_errors.loc[
                simulation_errors[
                    "number_of_agents"
                ]
                == number_of_centroid_agents
            ]
            .copy()
        )

        if simulation_errors.empty:
            print(
                f"No complete representative simulation for "
                f"machine {machine_id}, cluster {cluster_id}."
            )
            return None

        simulation_errors[
            "centroid_distance"
        ] = (
            simulation_errors[
                "centroid_objective"
            ]
            ** (1 / theta)
        )

        simulation_errors = (
            simulation_errors
            .sort_values(
                [
                    "centroid_objective",
                    "simulation_id",
                ],
                ascending=[True, True],
            )
            .reset_index(drop=True)
        )

        representative_simulation = (
            simulation_errors.iloc[0]
        )

        centroid_simulation_id = (
            representative_simulation[
                "simulation_id"
            ]
        )

        centroid_objective = float(
            representative_simulation[
                "centroid_objective"
            ]
        )

        centroid_error = float(
            representative_simulation[
                "centroid_distance"
            ]
        )

        # Retrieve valid discrete actions from the representative
        # historical simulation.
        centroid_rows = (
            selected_cluster_data.loc[
                selected_cluster_data[
                    "simulation_id"
                ]
                == centroid_simulation_id
            ]
            .sort_values("agent_id")
            .drop_duplicates(
                subset=["agent_id"],
                keep="first",
            )
        )

        if centroid_rows.empty:
            return None

        centroid_time_by_agent = {
            int(agent_id): float(centroid_time)
            for agent_id, centroid_time in zip(
                cluster_centroid["agent_id"],
                cluster_centroid[
                    "centroid_time"
                ],
            )
        }

        centroid_action_by_agent = {
            int(agent_id): int(action)
            for agent_id, action in zip(
                centroid_rows["agent_id"],
                centroid_rows["action"],
            )
        }

        # ========================================================
        # Construct the with-AV cluster baseline
        # ========================================================
        cluster_centroid_travel_times_list = []
        returned_agent_ids = []

        for agent_id in known_agent_ids:
            agent_id = int(agent_id)

            if agent_id not in centroid_time_by_agent:
                continue

            if agent_id not in centroid_action_by_agent:
                continue

            current_record = (
                current_records_by_agent.get(
                    agent_id
                )
            )

            if current_record is None:
                continue

            centroid_record = dc(
                current_record
            )

            centroid_record["action"] = int(
                centroid_action_by_agent[
                    agent_id
                ]
            )

            centroid_record["travel_time"] = float(
                centroid_time_by_agent[
                    agent_id
                ]
            )

            centroid_record.pop(
                "reward",
                None,
            )

            cluster_centroid_travel_times_list.append(
                centroid_record
            )

            returned_agent_ids.append(
                agent_id
            )

        if not cluster_centroid_travel_times_list:
            return None

        # The AV must be present in the with-AV baseline.
        if machine_id not in returned_agent_ids:
            print(
                f"Machine {machine_id} is missing from the "
                f"centroid baseline for cluster {cluster_id}."
            )
            return None

        # ========================================================
        # Remove the AV and rerun SUMO
        # ========================================================
        try:
            marginal_cost_result = (
                self.calculate_marginal_cost_for_agent_id(
                    [machine_id],
                    cluster_centroid_travel_times_list,
                )
            )
        except Exception as exc:
            print(
                f"Marginal-cost calculation failed for "
                f"machine {machine_id}, cluster {cluster_id}: "
                f"{exc}"
            )
            return None

        machine_marginal_cost = marginal_cost_result.get(
            machine_id,
            {},
        )

        if machine_marginal_cost is None:
            machine_marginal_cost = {}

        # Sum the travel-time differences immediately.
        # Do not save the individual affected-agent values.
        total_external_marginal_cost = float(
            sum(
                float(cost)
                for affected_agent_id, cost
                in machine_marginal_cost.items()
                if int(affected_agent_id) != machine_id
            )
        )

        return total_external_marginal_cost


    def calculate_approximated_marginal_cost(
        self,
        cluster_csv="agent_actions_with_cluster.csv",
        theta=2,
        acceptance_quantile=0.95,
        precompute_all_clusters=False,
        force_recompute=False,
    ):
        """
        Compute compatibility-weighted marginal costs using a cache.

        Cache structure
        ---------------
        {
            machine_id: {
                cluster_id: total_external_marginal_cost
            }
        }

        For every machine i and cluster c:

            MC[i, c] = sum over j != i of:
                T_with_i[j, c] - T_without_i[j, c]

        For the current episode:

            estimated_MC[i] =
                sum over compatible clusters c of:
                normalized_weight[i, c] * MC[i, c]

        Returns
        -------
        dict
            {
                machine_id: aggregated_marginal_cost
            }
        """

        if theta <= 0:
            raise ValueError(
                f"theta must be positive, received {theta}."
            )

        if not 0 < acceptance_quantile < 1:
            raise ValueError(
                "acceptance_quantile must be between 0 and 1."
            )

        cluster_data = pd.read_csv(cluster_csv)

        required_columns = {
            "simulation_id",
            "agent_id",
            "cluster",
            "travel_time",
            "action",
        }

        missing_columns = (
            required_columns - set(cluster_data.columns)
        )

        if missing_columns:
            raise ValueError(
                "The cluster CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if cluster_data.empty:
            raise ValueError(
                f"The cluster CSV '{cluster_csv}' is empty."
            )

        cluster_data["agent_id"] = pd.to_numeric(
            cluster_data["agent_id"],
            errors="raise",
        ).astype(int)

        cluster_data["cluster"] = pd.to_numeric(
            cluster_data["cluster"],
            errors="raise",
        ).astype(int)

        cluster_data["travel_time"] = pd.to_numeric(
            cluster_data["travel_time"],
            errors="raise",
        ).astype(float)

        cluster_data["action"] = pd.to_numeric(
            cluster_data["action"],
            errors="raise",
        ).astype(int)

        if not hasattr(
            self,
            "approximated_marginal_cost_cache",
        ):
            self.approximated_marginal_cost_cache = {}

        approximated_marginal_cost_matrix = {}
        results_by_machine = {}

        for machine in self.machine_agents:
            machine_id = int(machine.id)

            # ----------------------------------------------------
            # Current partial outcome for this AV
            # ----------------------------------------------------
            av_result = self.get_previous_actions(
                agents=self.travel_times_list,
                av_agent_id=machine_id,
            )

            known_agent_ids = [
                int(agent_id)
                for agent_id in av_result["agent_ids"]
            ]

            current_records_by_agent = {
                int(record["id"]): record
                for record in self.travel_times_list
            }

            # ----------------------------------------------------
            # Compute current cluster compatibilities
            # ----------------------------------------------------
            assignment = self.assign_joint_action_to_cluster(
                result=av_result,
                cluster_csv=cluster_csv,
                theta=theta,
                acceptance_quantile=acceptance_quantile,
            )

            ranking = assignment["ranking"].copy()

            ranking["cluster"] = (
                ranking["cluster"].astype(int)
            )

            compatible_ranking = (
                ranking.loc[
                    ranking["compatible"].astype(bool)
                ]
                .sort_values(
                    ["objective", "cluster"],
                    ascending=[True, True],
                )
                .reset_index(drop=True)
            )

            compatible_clusters = [
                int(cluster_id)
                for cluster_id in (
                    compatible_ranking["cluster"].tolist()
                )
            ]

            machine_cache = (
                self.approximated_marginal_cost_cache
                .setdefault(
                    machine_id,
                    {},
                )
            )

            # Precompute every ranked cluster or only those that
            # are compatible in the current episode.
            if precompute_all_clusters:
                clusters_to_prepare = (
                    ranking
                    .sort_values(
                        ["objective", "cluster"]
                    )
                    .drop_duplicates(
                        subset=["cluster"]
                    )
                    .reset_index(drop=True)
                )
            else:
                clusters_to_prepare = (
                    compatible_ranking
                )

            cache_hit_by_cluster = {}

            # ====================================================
            # Compute only missing cache values
            # ====================================================
            for _, cluster_row in (
                clusters_to_prepare.iterrows()
            ):
                cluster_id = int(
                    cluster_row["cluster"]
                )

                is_cached = (
                    cluster_id in machine_cache
                )

                if is_cached and not force_recompute:
                    cache_hit_by_cluster[
                        cluster_id
                    ] = True
                    continue

                cache_hit_by_cluster[
                    cluster_id
                ] = False

                cluster_total_marginal_cost = (
                    self
                    ._compute_cluster_total_marginal_cost(
                        machine_id=machine_id,
                        cluster_id=cluster_id,
                        known_agent_ids=known_agent_ids,
                        cluster_data=cluster_data,
                        current_records_by_agent=(
                            current_records_by_agent
                        ),
                        theta=theta,
                    )
                )

                if cluster_total_marginal_cost is None:
                    continue

                # Save only the scalar total.
                machine_cache[cluster_id] = float(
                    cluster_total_marginal_cost
                )

                """print(
                    f"Cached marginal cost for machine "
                    f"{machine_id}, cluster {cluster_id}: "
                    f"{machine_cache[cluster_id]}"
                )"""

            # ====================================================
            # Retrieve cached totals for compatible clusters
            # ====================================================
            raw_compatibilities = {}
            total_marginal_cost_by_cluster = {}

            for _, compatible_row in (
                compatible_ranking.iterrows()
            ):
                cluster_id = int(
                    compatible_row["cluster"]
                )

                if cluster_id not in machine_cache:
                    continue

                compatibility = float(
                    compatible_row["compatibility"]
                )

                if (
                    not np.isfinite(compatibility)
                    or compatibility < 0
                ):
                    compatibility = 0.0

                raw_compatibilities[
                    cluster_id
                ] = compatibility

                total_marginal_cost_by_cluster[
                    cluster_id
                ] = float(
                    machine_cache[cluster_id]
                )

            valid_clusters = list(
                total_marginal_cost_by_cluster.keys()
            )

            normalized_weights = {}
            weighted_contribution_by_cluster = {}
            aggregated_total_marginal_cost = 0.0

            # ====================================================
            # Normalize compatibility weights
            # ====================================================
            if valid_clusters:
                total_compatibility = float(
                    sum(
                        raw_compatibilities[
                            cluster_id
                        ]
                        for cluster_id in valid_clusters
                    )
                )

                if total_compatibility > 0:
                    normalized_weights = {
                        cluster_id: float(
                            raw_compatibilities[
                                cluster_id
                            ]
                            / total_compatibility
                        )
                        for cluster_id in valid_clusters
                    }
                else:
                    equal_weight = (
                        1.0 / len(valid_clusters)
                    )

                    normalized_weights = {
                        cluster_id: float(
                            equal_weight
                        )
                        for cluster_id in valid_clusters
                    }

                # =================================================
                # Weight the cached cluster totals
                # =================================================
                weighted_contribution_by_cluster = {
                    cluster_id: float(
                        normalized_weights[
                            cluster_id
                        ]
                        * total_marginal_cost_by_cluster[
                            cluster_id
                        ]
                    )
                    for cluster_id in valid_clusters
                }

                aggregated_total_marginal_cost = float(
                    sum(
                        weighted_contribution_by_cluster.values()
                    )
                )

            approximated_marginal_cost_matrix[
                machine_id
            ] = aggregated_total_marginal_cost

            # Current-episode diagnostics contain only scalar
            # cluster totals—no per-agent marginal costs.
            results_by_machine[machine_id] = {
                "compatible_clusters": (
                    compatible_clusters
                ),
                "used_clusters": (
                    valid_clusters
                ),
                "cached_clusters": sorted(
                    machine_cache.keys()
                ),
                "cache_hit_by_cluster": (
                    cache_hit_by_cluster
                ),
                "raw_cluster_compatibilities": (
                    raw_compatibilities
                ),
                "cluster_weights": (
                    normalized_weights
                ),
                "total_marginal_cost_by_cluster": (
                    total_marginal_cost_by_cluster
                ),
                "weighted_contribution_by_cluster": (
                    weighted_contribution_by_cluster
                ),
                "aggregated_total_marginal_cost": (
                    aggregated_total_marginal_cost
                ),
            }


        self.approximated_marginal_cost_details = (
            results_by_machine
        )

        #print("approximated marginal cost details", self.approximated_marginal_cost_details, "\n\n")

        return approximated_marginal_cost_matrix


    ##########################################
    ### support for MultiSyncDataCollector ###
    ##########################################

    def multisync_env_factories(self, env_wrapper, count: int = 0) -> list:
        """ Creates array of factories that create environments identical to this one. Intended to be used with MultiSyncDataCollector from torchrl. It is assumed that each episode is one day long and human do not learn.
            
            Args:
                env_wrapper (Callable): Callable used for wrapping the environment. Should take ``env`` as an argument and return wrapped ``env``. Put all PettingZoo wrappers inside it.
                count (int): Number of factories to be returned.
        """
        if 0 == count:
            count = os.cpu_count()-1
        
        manager = Manager()
        shared_ns = manager.Namespace()
        shared_ns.episode = self.day
        lock = manager.Lock()
        counter = MultiSyncTrafficEnvironment._EpisodeCounterDescriptor(shared_ns, lock)
        counter.inject(self, "day", readonly = False)
        counter.inject(self.simulator, "runs", readonly = True)

        def make_make_env(i):
            agents      = dc(self.all_agents)
            seed        = self.seed
            params      = dc(self.kwargs)
            sim_params  = params[kc.SIMULATOR]
            sim_params[kc.USE_LIBSUMO] = True
            plotter_params = params[kc.PLOTTER]
            plotter_params[kc.CLEAR_RECORDS] = False


            def make_env():
                env = MultiSyncTrafficEnvironment(
                    seed            = seed,
                    create_agents   = False,
                    create_paths    = False,
                    agents          = agents,
                    **params
                )

                counter.inject(env, "day", readonly = False)
                counter.inject(env.simulator, "runs", readonly = True)

                env.start()
                env.human_learning = False
                return env_wrapper(env)
            return make_env
        return [make_make_env(i) for i in range(count)]



class MultiSyncTrafficEnvironment(TrafficEnvironment):
    def close(self) -> None:
        self.stop_simulation()


    class _EpisodeCounterDescriptor:
        """ Shared (across both classes and processes) episode counter. Intended to make episode data consistent across many workers. 

        Args:
            manager (multiprocessing.Manager): Manager for episode variable.
            lock (multiprocessing.Lock): Lock associated with the variable. Note: ``manager.Lock()`` is a good candidate.
            monotone (bool): If set to ``True`` the counter does not allow the value to be decreased.
        """
        def __init__(self, manager, lock, monotone: bool = False):
            with lock:
                self.value = manager.episode
            self.manager = manager
            self.lock = lock
            self.monotone = monotone
            self.ro = set()
            with lock:
                self.value = manager.episode

        def __get__(self, obj, objtype=None):
            return self.value

        def __set__(self, obj, new_value):
            if obj in self.ro:
                return

            delta = new_value - self.value
            if self.monotone and delta < 0:
                return

            with self.lock:
                self.manager.episode += delta
                self.value = self.manager.episode

        def inject(self, obj, field_name, readonly: bool = True) -> None:
            """ Inject a shared episode counter into object. 

            Args:
                obj (Object): Object the counter will be injected into.
                field_name (str): Name of the field to be shadowed by the injected counter.
                readonly (bool): If set to ``True``, then counter value cannot be changed from ``obj``. 
            Returns:
                None
            """

            old_cls = obj.__class__
            new_cls = new_class(
                "_CounterInjected" + old_cls.__name__.replace("_", ""),
                (old_cls,),
                kwds=None,
                exec_body = lambda ns: ns.update ({ field_name: self }),
            )

            obj.__class__ = new_cls
            if readonly:
                self.ro.add(obj)


class _MarginalCostWorker:
    @staticmethod
    def initWorker(all_agents, travel_times_list, sumo_seed, machines_to_all, kwargs):
        global _worker_all_agents
        global _worker_travel_times_list
        global _worker_sumo_seed
        global _worker_machines_to_all
        global _worker_kwargs
        global _worker_env
        _worker_all_agents          = all_agents
        _worker_travel_times_list   = travel_times_list
        _worker_sumo_seed           = sumo_seed
        _worker_machines_to_all     = machines_to_all
        _worker_kwargs              = kwargs

        # mock agents used for environment initialization
        initial_agents = []
        for agent in all_agents:
            initial_agents.append(OneActionAgent(agent.id, agent.start_time, agent.origin, agent.destination, agent.last_action))

        initial_agents.pop() #during marginal calculation one agents is always gone

        # Init environment
        params = kwargs
        sim_params  = params[kc.SIMULATOR]
        #sim_params[kc.USE_LIBSUMO] = True
        # Libsumo is failing to load locally on Windows.
        # Use the standard TraCI client instead.
        if sys.platform == "win32":
            sim_params[kc.USE_LIBSUMO] = False
        else:
            sim_params[kc.USE_LIBSUMO] = True
        sim_params[kc.USE_SUMO_TELEPORT] = True
        sim_params[kc.DISABLE_SUMO_STATS] = True
        plotter_params = params[kc.PLOTTER]
        plotter_params[kc.CLEAR_RECORDS] = False
        plotter_params[kc.RECORD] = False
        _worker_env = TrafficEnvironment(
            seed=sumo_seed, 
            create_agents=False, 
            create_paths=False, 
            agents=initial_agents, 
            **params
        )
        _worker_env._record = _MarginalCostWorker.noop
        _worker_env.simulator.records_folder = os.devnull
        _worker_env.start()
    
    @staticmethod
    def filter_agents(env, agent_id):
        all_agents = _worker_all_agents

        # Filter agents
        filtered_agents = []
        for agent in all_agents:
            if agent.id == agent_id:
                continue

            filtered_agents.append(OneActionAgent(agent.id, agent.start_time, agent.origin, agent.destination, agent.last_action))

        env.all_agents      = filtered_agents
        env.human_agents    = filtered_agents

    @staticmethod
    def task(agent_id):
        all_agents          = _worker_all_agents
        travel_times_list   = _worker_travel_times_list
        sumo_seed           = _worker_sumo_seed
        machines_to_all     = _worker_machines_to_all
        kwargs              = _worker_kwargs
        env                 = _worker_env
        human_parameters    = kwargs[kc.AGENTS][kc.HUMAN_PARAMETERS]

        _MarginalCostWorker.filter_agents(env, agent_id)
        env.reset()
        env.step()
        
        agent_marginal_cost = {}
        for agent in all_agents:
            if machines_to_all == False: #whether to calculate the impact of deleting a machine agent on the other machine agents
                                         #or on human agents as well
                if agent.id == agent_id or agent.kind == kc.TYPE_HUMAN:
                    if agent.kind != kc.TYPE_HUMAN:
                        agent_marginal_cost[agent.id] = 0.0 
                    continue

            after_step_time = _MarginalCostWorker.get_travel_time_by_id(env.last_episode_travel_times, agent.id)
            initial_time = _MarginalCostWorker.get_travel_time_by_id(travel_times_list, agent.id)

            if initial_time is not None and after_step_time is not None:
                difference = initial_time - after_step_time  #consistent with the formulation from Anastasias paper
                #print(f"{difference}")
                agent_marginal_cost[agent.id] = difference
            else:
                agent_marginal_cost[agent.id] = 0.0

        #print(agent_marginal_cost)
        return agent_marginal_cost


    @staticmethod
    def get_travel_time_by_id(travel_times_list, agent_id):
        for entry in travel_times_list:
            if entry['id'] == agent_id:
                return entry['travel_time']
        return None


    @staticmethod
    def noop(*args, **kwargs):
        pass

