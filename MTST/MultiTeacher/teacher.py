import os
import time
import logging
import random
import grid2op
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend
from grid2op.Exceptions import *
from MTST.Utils.logger import logging


# Configure the logger
logging.basicConfig(
    filename='generation_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class Teacher:
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.env = grid2op.make(self.config.env_name, reward_class=L2RPNSandBoxScore,
                                backend=LightSimBackend())

        self.DATA_PATH = self.env.get_path_env() #'C:\\Users\\Ernest\\data_grid2op\\l2rpn_wcci_2022'  # for demo only, use your own dataset
        self.SCENARIO_PATH = self.env.chronics_handler.path #'C:\\Users\\Ernest\\data_grid2op\\l2rpn_wcci_2022'
        if kwargs.get('save_path'):
            self.SAVE_PATH = kwargs.get('save_path')
        else:
            self.SAVE_PATH = Path("MTST\\data")



        self.NUM_EPISODES = self.config.num_episode  # each scenario runs 100 times for each attack (or to say, sample 100 points)



    def get_sub_actions(self, substations:list):
        all_actions = [self.env.action_space({})]
        for sub in substations:
            topo = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space, sub)
            for act in topo:
                all_actions.append(act)
            
        return all_actions

    

    def topology_search(self, dst_step, substations:list):
        obs = self.env.get_obs()
        min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
        logging.info("step-%s, line-%s(from bus-%d to bus-%d) overflows, max rho is %.5f" %
            (dst_step, overflow_id, self.env.line_or_to_subid[overflow_id],
            self.env.line_ex_to_subid[overflow_id], obs.rho.max()))
        #all_actions = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space)
        sub_actions = self.get_sub_actions(substations)
        action_chosen = self.env.action_space({})
        tick = time.time()
        for action in sub_actions:
            if not self.env._game_rules(action, self.env):
                continue
            obs_, _, done, _ = obs.simulate(action)
            if (not done) and (obs_.rho.max() < min_rho):
                min_rho = obs_.rho.max()
                action_chosen = action
        print("find a greedy action and max rho decreases to %.5f, search duration: %.2f" %
            (min_rho, time.time() - tick))
        return action_chosen



    def get_substation_connections(self):
        """
        Returns a dictionary with each substation ID and the number of powerlines connected to it.

        Parameters:
        - env: Grid2Op environment instance

        Returns:
        - substation_connections: Dictionary with substation ID as key and connection count as value
        """
        # Initialize a dictionary to store the number of connections for each substation
        substation_connections = defaultdict(int)

        # Retrieve powerline-to-substation mappings
        line_or_to_subid = self.env.line_or_to_subid  # Array of origin substations for each powerline
        line_ex_to_subid = self.env.line_ex_to_subid  # Array of extremity substations for each powerline

        # Count connections for each substation
        for line_id in range(self.env.n_line):
            origin_substation = line_or_to_subid[line_id]
            extremity_substation = line_ex_to_subid[line_id]
            
            # Increment the connection count for each substation
            substation_connections[origin_substation] += 1
            substation_connections[extremity_substation] += 1

        return substation_connections
    



    def find_most_connected_substations(self, substation_connections, top_n=5):
        """
        Finds the top N substations with the highest number of powerline connections.

        Parameters:
        - substation_connections: Dictionary with substation ID as key and connection count as value
        - top_n: Number of top substations to return

        Returns:
        - List of tuples (substation_id, connection_count) sorted by connection count in descending order
        """
        # Sort substations by the number of connections in descending order
        sorted_substations = sorted(substation_connections.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_substations[:top_n]
    


    def find_powerlines_connected_to_substations(self, target_substations):
        """
        Finds all powerline IDs that are connected to the given substations.

        Parameters:
        - env: Grid2Op environment instance
        - target_substations: List of substation IDs to check for connections

        Returns:
        - connected_powerlines: Dictionary where each substation ID is a key and 
        the value is a list of powerline IDs connected to that substation
        """
        connected_powerlines = {sub: [] for sub in target_substations}

        # Retrieve powerline-to-substation mappings
        line_or_to_subid = self.env.line_or_to_subid  # Origin substation for each powerline
        line_ex_to_subid = self.env.line_ex_to_subid  # Extremity substation for each powerline

        # Loop through each powerline to find connections to target substations
        for line_id in range(self.env.n_line):
            origin_substation = line_or_to_subid[line_id]
            extremity_substation = line_ex_to_subid[line_id]

            # Check if origin or extremity matches any target substation
            if origin_substation in target_substations:
                connected_powerlines[origin_substation].append(line_id)
            if extremity_substation in target_substations:
                connected_powerlines[extremity_substation].append(line_id)

        return connected_powerlines



    def line2attack(self):
        substation_connections = self.get_substation_connections()
        top_substations = self.find_most_connected_substations(substation_connections, top_n=10)
        target_substations = [item[0] for item in top_substations]

        LINES2ATTACK = []
        # Find powerlines connected to the target substations
        result = self.find_powerlines_connected_to_substations(target_substations)

        # Print the result
        for substation, lines in result.items():
            for i in lines:
                LINES2ATTACK.append(i)
        
        return LINES2ATTACK
    


    def save_sample(self, obs, action, obs_, dst_step, line_to_disconnect):
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        if action == self.env.action_space({}):
            return None  # not necessary to save a "do nothing" action
        act_or, act_ex, act_gen, act_load = [], [], [], []
        for key, val in action.as_dict()['set_bus_vect'][
            action.as_dict()['set_bus_vect']['modif_subs_id'][0]].items():
            if val['type'] == 'line (extremity)':
                act_ex.append(key)
            elif val['type'] == 'line (origin)':
                act_or.append(key)
            elif val['type'] == 'load':
                act_load.append(key)
            else:
                act_gen.append(key)
        pd.concat(
            (
                pd.DataFrame(
                    np.array(
                        [
                            self.env.chronics_handler.get_name(), 
                            dst_step, 
                            line_to_disconnect,
                            self.env.line_or_to_subid[line_to_disconnect],
                            self.env.line_ex_to_subid[line_to_disconnect], 
                            str(np.where(obs.rho > 1)[0].tolist()),  # Convert list to string
                            str([i for i in np.around(obs.rho[np.where(obs.rho > 1)], 2)]),  # Convert list to string
                            action.as_dict()['set_bus_vect']['modif_subs_id'][0], 
                            str(act_or),  # Convert list to string
                            str(act_ex),  # Convert list to string
                            str(act_gen),  # Convert list to string
                            str(act_load),  # Convert list to string
                            float(obs.rho.max()),  # Ensure numeric values are floats
                            int(obs.rho.argmax()),  # Ensure indexes are integers
                            float(obs_.rho.max()),  # Ensure numeric values are floats
                            int(obs_.rho.argmax())  # Ensure indexes are integers
                        ]
                    ).reshape([1, -1])
                ),
                pd.DataFrame(np.concatenate((obs.to_vect(), obs_.to_vect(), action.to_vect())).reshape([1, -1]))
            ),
            axis=1
        ).to_csv(os.path.join(self.SAVE_PATH, 'Experiences1.csv'), index=0, header=0, mode='a')


    def generate(self, LINES2ATTACK, substations: list):
        logging.info("Start Generation")
        for episode in range(self.NUM_EPISODES):
            # traverse all attacks
            for line_to_disconnect in LINES2ATTACK:
                try:
                    # if lightsim2grid is available, use it.
                    from lightsim2grid import LightSimBackend
                    backend = LightSimBackend()
                    env = grid2op.make(dataset=self.DATA_PATH, chronics_path=self.SCENARIO_PATH, backend=backend)
                except:
                    env = grid2op.make(dataset=self.DATA_PATH, chronics_path=self.SCENARIO_PATH)

                env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
                
                # traverse all scenarios
                for chronic in range(len(os.listdir(self.SCENARIO_PATH))):
                    try:
                        env.reset()
                        dst_step = episode * 72 + random.randint(0, 72)  # a random sampling every 6 hours
                        logging.info('\n\n' + '*' * 50 + '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                            env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                            env.line_or_to_subid[line_to_disconnect], env.line_ex_to_subid[line_to_disconnect]))
                        
                        # to the destination time-step
                        env.fast_forward_chronics(dst_step - 1)
                        
                        # Perform the initial "do nothing" action
                        try:
                            obs, reward, done, _ = env.step(env.action_space({}))
                        except Grid2OpException as e:
                            print(f"Grid2OpException during initial step: {e}")
                            continue

                        if done:
                            break

                        # disconnect the targeted line
                        new_line_status_array = np.zeros(obs.rho.shape, dtype=np.int32)
                        new_line_status_array[line_to_disconnect] = -1
                        action = env.action_space({"set_line_status": new_line_status_array})

                        try:
                            obs, reward, done, _ = env.step(action)
                        except Grid2OpException as e:
                            print(f"Grid2OpException during line disconnection step: {e}")
                            continue

                        if obs.rho.max() < 1:
                            # not necessary to do a dispatch
                            continue
                        else:
                            # search a greedy action
                            try:
                                action = self.topology_search(dst_step=dst_step, substations=substations)
                                obs_, reward, done, _ = env.step(action)
                                self.save_sample(obs, action, obs_, dst_step, line_to_disconnect)
                            except Grid2OpException as e:
                                print(f"Grid2OpException during greedy action step: {e}")
                                continue

                    except Exception as e:
                        print(f"Exception during scenario handling: {e}")
                        continue

        logging.info(f"""\n\n #########################################\n\n
                \t   SEARCH IS DONE    \n\n #########################################""")

        print(f"""\n\n #########################################\n\n
                \t   SEARCH IS DONE    \n\n #########################################""")

    """def generate(self, LINES2ATTACK, substations:list):
        for episode in range(self.NUM_EPISODES):
            # traverse all attacks
            for line_to_disconnect in LINES2ATTACK:
                try:
                    # if lightsim2grid is available, use it.
                    from lightsim2grid import LightSimBackend
                    backend = LightSimBackend()
                    env = grid2op.make(dataset=self.DATA_PATH, chronics_path=self.SCENARIO_PATH, backend=backend)
                except:
                    env = grid2op.make(dataset=self.DATA_PATH, chronics_path=self.SCENARIO_PATH)
                env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
                # traverse all scenarios
                for chronic in range(len(os.listdir(self.SCENARIO_PATH))):
                    env.reset()
                    dst_step = episode * 72 + random.randint(0, 72)  # a random sampling every 6 hours
                    print('\n\n' + '*' * 50 + '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                        env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                        env.line_or_to_subid[line_to_disconnect], env.line_ex_to_subid[line_to_disconnect]))
                    # to the destination time-step
                    env.fast_forward_chronics(dst_step - 1)
                    obs, reward, done, _ = env.step(env.action_space({}))
                    if done:
                        break
                    # disconnect the targeted line
                    new_line_status_array = np.zeros(obs.rho.shape, dtype=np.int32)
                    new_line_status_array[line_to_disconnect] = -1
                    action = env.action_space({"set_line_status": new_line_status_array})
                    obs, reward, done, _ = env.step(action)
                    if obs.rho.max() < 1:
                        # not necessary to do a dispatch
                        continue
                    else:
                        # search a greedy action
                        action = self.topology_search(dst_step=dst_step, substations=substations)
                        obs_, reward, done, _ = env.step(action)
                        self.save_sample(obs, action, obs_, dst_step, line_to_disconnect)"""