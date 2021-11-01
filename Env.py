import numpy as np
import time
import copy
from sklearn.preprocessing import MinMaxScaler

class Layout:
    def __init__(self, num_isl, k, num_tpark, num_pass = 1, num_stack=10, spot_size = (2.5, 5), clear_way = 3):
        # num_isl   - number of parking island (int)
        # k         - the value of k in each k-stack. Every island contain 2k columns. (list, size = isl_num)
        # num_tpark - the number of temporary parking lane in each gap. (list, size = isl_num + 1)
        # num_pass  - the number of passing lane in each gap. (list, size = isl_num + 1, default 1, means all gaps contain only 1 passing lane)
        # num_stack - the number of k-stack in each island. (int, default 10)
        # spot_size - the size of parking spot. (tuple, default (2, 5))
        # clear_way - way for getting into / out the facility. (int, only need width, defalut 3)
        isl_info = []
        dwell_info = []
        arrive_info = []
        for isl_ind in range(num_isl):
            isl_info.append(np.zeros([num_stack, 2 * k[isl_ind]]))
            dwell_info.append(np.zeros([num_stack, 2 * k[isl_ind]]))
            arrive_info.append(np.zeros([num_stack, 2 * k[isl_ind]]))
        tpark_info = []
        tpark_capacity = int(np.floor(num_stack * spot_size[0] / spot_size[1]))
        if len(num_tpark) != num_isl + 1:
            raise ValueError("Please input the right num_tpark")
        for tpark_ind in range(len(num_tpark)):
            tpark_info.append(np.zeros([tpark_capacity, int(num_tpark[tpark_ind])]))
        pass_info = []
        pass_capacity = int(np.floor(num_stack * spot_size[0] / spot_size[1]))
        if isinstance(num_pass, int):
            num_pass = [num_pass for i in range(num_isl + 1)]
        if len(num_pass) != num_isl + 1:
            raise ValueError("Please input the right num_pass")
        for pass_ind in range(len(num_pass)):
            pass_info.append(np.zeros([pass_capacity,num_pass[pass_ind]]))
        self.isl_info = isl_info
        self.dwell_info = dwell_info
        self.tpark_info = tpark_info
        self.pass_info = pass_info
        self.spot_size = spot_size
        self.clear_way = clear_way
        self.arrive_info = arrive_info


    def get_size(self):
        # get the parking facility's overall size
        tpark_x = self.spot_size[0] * sum([tpark.shape[1] for tpark in self.tpark_info])
        pass_x = self.spot_size[0] * sum([pass_lane.shape[1] for pass_lane in self.pass_info])
        island_x = self.spot_size[1] * sum([island.shape[1] for island in self.isl_info])
        try:
            whole_x = tpark_x + pass_x + island_x
            whole_y = self.clear_way + self.isl_info[0].shape[0] * self.spot_size[0]
            return whole_x, whole_y
        except:
            print(self.isl_info)
            raise ValueError("list index out of range")


    def get_capacity(self):
        return sum(island.shape[0] * island.shape[1] for island in self.isl_info)


class Simulation:
    def __init__(self, event, ind, demand, dwell_type, k, stack):
        self.k = k
        self.stack = stack
        self.facility = Layout(len(k), k, [1 for i in range(len(k) + 1)],num_stack=stack)
        self.event, self.ind, self.demand, self.dwell_type = event, ind, demand, dwell_type
        self.copy_event, self.copy_ind, self.copy_demand, self.copy_dwell_type = copy.deepcopy(event), copy.deepcopy(ind), copy.deepcopy(demand), copy.deepcopy(dwell_type)
        self.current_pointer = 0
        self.sim_relocate = 0
        self.sim_block = 0
        self.sim_reject = 0
        self.transformer = MinMaxScaler()
        self.layout_space = (2, stack, 2 * sum(k))
        self.veh_space = 2
        self.action_space = len(k) * stack


    def arrive(self, veh, dwell, event, action):
        # 首先判断当前veh是否是tpark中的
        for i in range(len(self.facility.tpark_info)):
            veh_arg = np.argwhere(self.facility.tpark_info[i] == veh)
            if veh_arg.size > 0:
                veh_arg = veh_arg[0]
                self.facility.tpark_info[i][veh_arg[0]][veh_arg[1]] = 0
        flag = 0
        isl = int(action[0])
        stack = int(action[1])
        col = None
        if np.argwhere(self.facility.isl_info[isl][stack] == 0).size > 0:
            col_ls = np.argwhere(self.facility.isl_info[isl][stack] == 0)
            middle_index = (len(self.facility.isl_info[isl][stack]) + 1) // 2
            col = col_ls[np.argmin(abs(col_ls - middle_index))]
            self.facility.isl_info[isl][stack][col] = veh
            self.facility.dwell_info[isl][stack][col] = dwell
            self.facility.arrive_info[isl][stack][col] = event
        else:
            flag = "reject"
        return flag, isl, stack, col


    def depart(self, veh):
        # veh: the target vehilce which wanna go out
        # facility: the parking faiclity (layout object)
        # First, locate the veh
        facility = self.facility
        relocate = 0
        search_turn = 0
        block = 0
        # 首先判断该到达车辆是否是tpark中重定位的车辆
        for i in range(len(facility.tpark_info)):
            if np.argwhere(facility.tpark_info[i] == veh).size > 0:
                # 如果在tpark中找到了该车，由于车辆排列，不可能出现block，因此直接清空车位出场。
                veh_arg = np.argwhere(facility.tpark_info[i] == veh)[0]
                facility.tpark_info[i][veh_arg[0]][veh_arg[1]] = 0
                relocate_veh = np.array([])
                return relocate_veh, relocate, block
        for i in range(len(facility.isl_info)):
            island =  facility.isl_info[i]
            if np.argwhere(island == veh).size > 0:
                locate = np.argwhere(island == veh)[0]
                stack_ind = locate[0]
                spot_ind = locate[1]
                stack_2k = len(island[stack_ind])
                stack_k = len(island[stack_ind]) / 2
                if spot_ind == 0 or spot_ind == stack_2k - 1:
                    # 如果在最外侧，直接走
                    island[stack_ind][spot_ind] = 0
                    relocate_veh = np.array([])
                    return relocate_veh, relocate, block
                elif spot_ind <= stack_k - 1:
                    # 如果在左侧栈堆
                    front = island[stack_ind][ : spot_ind]
                    back = island[stack_ind][spot_ind + 1: ]
                    front_copy = front.copy()
                    back_copy = back.copy()
                    if len(np.where(front != 0)[0]) <= len(np.where(back != 0)[0]):
                        # 如果前方车辆比后面少，从前面走
                        if len(np.where(front_copy != 0)[0]) != 0:
                            # 如果有车辆重定位，则先将重定位车辆安放到临时停车道
                            # 检索临时停车道是否有空闲位置 i
                            # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                            if facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1] >= len(np.where(front_copy != 0)[0]):
                                args = np.argwhere(front_copy != 0)
                                relocate_veh = front_copy[args]
                                island[stack_ind][ : spot_ind] = [0 for i in range(len(front_copy))]
                                row = facility.tpark_info[i].shape[1]
                                lane = 0
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                island[stack_ind][spot_ind] = 0 # 目标车辆离场
                                return relocate_veh, relocate, block
                            # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                            else:
                                tpark_size = facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1]
                                args = np.argwhere(front_copy != 0)
                                relocate_veh = front_copy[args][: tpark_size]
                                for r in relocate_veh:
                                    relocate_args = np.argwhere(island[stack_ind] == r)
                                    island[stack_ind][relocate_args] = 0
                                row = facility.tpark_info[i].shape[1]
                                lane = 0
                                # 首先将可以移动的车辆移动
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                                block = 1
                                return relocate_veh, relocate, block
                        else:
                            # 如果没有车辆被重定位
                            args = np.argwhere(front_copy != 0)
                            relocate_veh = front_copy[args] # 此时为空
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                        return relocate_veh, relocate, block
                    else:
                        # 如果后方车辆比前面少，从后面走，此时重定位移动到的是i+1 tpark
                        if len(np.where(back_copy != 0)[0]) != 0:
                            # 如果后方车辆存在，则需要进行重定位
                            # 检索临时停车道是否有空闲位置
                            # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                            if facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1] >= len(np.where(back_copy != 0)[0]):
                                args = np.argwhere(back_copy != 0)
                                relocate_veh = back_copy[args]
                                island[stack_ind][spot_ind + 1 : ] = [0 for i in range(len(back_copy))]
                                row = facility.tpark_info[i + 1].shape[1]
                                lane = 0
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i + 1][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                island[stack_ind][spot_ind] = 0 # 目标车辆离场
                                return relocate_veh, relocate, block
                            # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                            else:
                                args = np.argwhere(back_copy != 0)
                                tpark_size = facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1]
                                relocate_veh = back_copy[args][len(back_copy[args]) - 1 - tpark_size : ]
                                for r in relocate_veh:
                                    relocate_args = np.argwhere(island[stack_ind] == r)
                                    island[stack_ind][relocate_args] = 0
                                row = facility.tpark_info[i + 1].shape[1]
                                lane = 0
                                # 首先将可以移动的车辆移动
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i + 1][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                                block = 1
                                return relocate_veh, relocate, block
                        else:
                            # 如果没有车辆被重定位
                            args = np.argwhere(back_copy != 0)
                            relocate_veh = back_copy[args] # 此时为空
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                            return relocate_veh, relocate, block
                elif spot_ind >= stack_k:
                    # 如果在右侧栈堆
                    back = island[stack_ind][ : spot_ind]
                    front = island[stack_ind][spot_ind + 1: ]
                    front_copy = front.copy()
                    back_copy = back.copy()
                    if len(np.where(front != 0)[0]) <= len(np.where(back != 0)[0]):
                        # 如果前方车辆比后面少，从前面走
                        if len(np.where(front_copy != 0)[0]) != 0:
                            # 如果有车辆重定位，则先将重定位车辆安放到临时停车道
                            # 检索临时停车道是否有空闲位置,此时是i + 1tpark
                            # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                            args = np.argwhere(front_copy != 0)
                            if facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1] >= len(np.where(front_copy != 0)[0]):
                                relocate_veh = front_copy[args]
                                island[stack_ind][spot_ind + 1 :] = [0 for i in range(len(front_copy))]
                                row = facility.tpark_info[i + 1].shape[1]
                                lane = 0
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i + 1][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                island[stack_ind][spot_ind] = 0 # 目标车辆离场
                                return relocate_veh, relocate, block
                            # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                            else:
                                args = np.argwhere(front_copy != 0)
                                tpark_size = facility.tpark_info[i + 1].shape[0] * facility.tpark_info[i + 1].shape[1]
                                relocate_veh = front_copy[args][len(front_copy[args]) - tpark_size : ]
                                for r in relocate_veh:
                                    relocate_args = np.argwhere(island[stack_ind] == r)
                                    island[stack_ind][relocate_args] = 0
                                row = facility.tpark_info[i + 1].shape[1]
                                lane = 0
                                # 首先将可以移动的车辆移动
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i + 1][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                                block = 1
                                return relocate_veh, relocate, block
                        else:
                            # 如果没有车辆被重定位
                            args = np.argwhere(front_copy != 0)
                            relocate_veh = front_copy[args] # 此时为空
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                            return relocate_veh, relocate, block
                    else:
                        # 如果后方车辆比前面少，从后面走，此时重定位移动到的是i tpark
                        if len(np.where(back_copy != 0)[0]) != 0:
                            # 如果后方车辆存在，则需要进行重定位
                            # 检索临时停车道是否有空闲位置
                            # 如果有足够的空闲位置，则重定位车辆在其中按顺序停放
                            if facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1] >= len(np.where(back_copy != 0)[0]):
                                args = np.argwhere(back_copy != 0)
                                relocate_veh = back_copy[args]
                                island[stack_ind][ : spot_ind] = [0 for i in range(len(back_copy))]
                                row = facility.tpark_info[i].shape[1]
                                lane = 0
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                island[stack_ind][spot_ind] = 0 # 目标车辆离场
                                return relocate_veh, relocate, block
                            # 如果没有足够的空闲位子，先移动可以移动的车辆，返回该部分移动过的车辆，并且没有移动过的车辆&需要离场的车辆
                            else:
                                args = np.argwhere(back_copy != 0)
                                tpark_size = facility.tpark_info[i].shape[0] * facility.tpark_info[i].shape[1]
                                relocate_veh = back_copy[args][tpark_size : ]
                                for r in relocate_veh:
                                    relocate_args = np.argwhere(island[stack_ind] == r)
                                    island[stack_ind][relocate_args] = 0
                                row = facility.tpark_info[i].shape[1]
                                lane = 0
                                # 首先将可以移动的车辆移动
                                for j in range(0, relocate_veh.size, row):
                                    count = 0
                                    while count < row and j + count < relocate_veh.size:
                                        veh_arg = j + count
                                        this_veh = relocate_veh[veh_arg][0]
                                        facility.tpark_info[i][lane][count] = this_veh
                                        count += 1
                                    lane += 1
                                relocate += len(relocate_veh)
                                # 然后返回block标识，如果有block标识，则event中首先加入relocate_veh的内容，然后relocate后，block_veh再离场
                                block = 1
                                return relocate_veh, relocate, block
                        else:
                            # 如果没有车子被重定位
                            args = np.argwhere(back_copy != 0)
                            relocate_veh = back_copy[args] # 此时为空
                            island[stack_ind][spot_ind] = 0 # 目标车辆离场
                            return relocate_veh, relocate, block
            # 如果车辆不在当前停车岛，继续寻找下一个停车岛
            search_turn += 1
        if search_turn >= len(facility.isl_info):
            # 如果找遍了所有停车岛都没有找到目标车辆
            # 先从tpark中找
            count = 0
            relocate = 0
            block = 0
            relocate_veh = np.array([])
            for i in range(len(facility.tpark_info)):
                if np.argwhere(facility.tpark_info[i] == veh).size > 0:
                    # 如果在tpark中找到了该车，由于车辆排列，不可能出现block，因此直接清空车位离场。
                    veh_arg = np.argwhere(facility.tpark_info[i] == veh)
                    facility.tpark_info[i][veh_arg][0] = 0
                    return relocate_veh, relocate, block
                count += 1
            if count >= len(facility.tpark_info):
                print(veh, facility.tpark_info, facility.isl_info)
                raise ValueError("Cannot find this vehicle:", veh)

    def reset(self):
        self.current_pointer = 0
        self.ind, self.event, self.demand, self.dwell_type = self.copy_ind, self.copy_event, self.copy_demand, self.copy_dwell_type
        self.facility = Layout(len(self.k), self.k, [1 for i in range(len(self.k) + 1)],num_stack=self.stack)
        self.sim_block = 0
        self.sim_reject = 0
        self.sim_relocate = 0
        current_dwell = self.dwell_type[0]
        current_event = self.event[0]
        isl_info = np.hstack(self.facility.isl_info)
        arrive_info = np.hstack(self.facility.arrive_info)
        dwell_info = np.hstack(self.facility.dwell_info)
        # arrive_info = self.transformer.fit_transform(arrive_info)
        # dwell_info = self.transformer.fit_transform(dwell_info)
        veh_info = np.array([current_event, current_dwell])
        veh_info = veh_info.reshape(2,1)
        # veh_info = self.transformer.fit_transform(veh_info)
        layout_info = np.stack([arrive_info, dwell_info], axis=0)
        veh_info = veh_info.reshape(2)
        state = [layout_info, veh_info]
        return state

    def step(self, action):
        start_time = time.time()
        reject = 0
        relocate = 0
        block = 0
        end = False
        fixed_pointer = copy.deepcopy(self.current_pointer)
        initial_ind = copy.deepcopy(self.ind)
        initial_event = copy.deepcopy(self.event)
        position = []
        v = self.ind[self.current_pointer]
        e = self.event[self.current_pointer]
        relocate_dwell = 0
        if v <= self.demand:
            # 入场情景，执行一次车位分配动作
            d = self.dwell_type[v - 1]
            flag, isl, stack, col = self.arrive(v,d,e,action)
            if flag != "reject":
                position = [isl, stack, col]
            if flag == "reject":
                reject += 1
                self.sim_reject += 1
                depart_index = np.argwhere(self.ind == v + self.demand)[0][0]
                self.ind = np.delete(self.ind, depart_index)
                self.event = np.delete(self.event, depart_index)
            self.current_pointer += 1
            v = self.ind[self.current_pointer]
            e = self.event[self.current_pointer]
        while v > self.demand and self.current_pointer < len(self.event):
            # 出场情景，清空所有需要出场的车辆
            relocate_veh, relocate_flag, block_flag = self.depart(v - self.demand)
            relocate += relocate_flag
            self.sim_relocate += relocate_flag
            block += block_flag
            self.sim_block += block_flag
            if relocate_flag != 0:
                relocate_veh = relocate_veh.flatten()
                relocate_veh = [int(i) for i in relocate_veh]
                relocate_dwell = sum([self.dwell_type[v - 1] for v in relocate_veh])
                relocate_event = [initial_event[int(i)] for i in relocate_veh]
                self.event = np.insert(self.event, self.current_pointer + 1, relocate_event)
                self.ind = np.insert(self.ind, self.current_pointer + 1, relocate_veh)
                if block_flag != 0:
                    self.event = np.insert(self.event, self.current_pointer + 1 + len(relocate_veh), e)
                    self.ind = np.insert(self.ind, self.current_pointer + 1 + len(relocate_veh), v)
            if self.current_pointer < len(self.event) - 1:
                self.current_pointer += 1
                v = self.ind[self.current_pointer]
                e = self.event[self.current_pointer]
            spare_spot = sum([len(np.argwhere(self.facility.isl_info[i] == 0)) for i in range(len(self.facility.isl_info))])
            if self.facility.get_capacity() - spare_spot == 0:
                end = True
                break
        if v > self.demand and end is False:
            raise ValueError("Error:", v, self.facility.isl_info)
        current_veh = initial_ind[fixed_pointer]
        current_dwell = self.dwell_type[current_veh - 1]
        current_event = initial_event[fixed_pointer]
        avg_salary = 30
        # if reject == 0:
        #     reward = (current_dwell * 3 - 3 * relocate_dwell - self.sim_relocate / 10) / 4
        # else:
        #     reward = (- current_dwell * 3 - avg_salary) / 4
        if reject == 0:
            if relocate == 0:
                reward = (self.demand / 3 - self.sim_reject - self.sim_relocate) / 10
            else:
                reward = - self.sim_relocate / 10
        else:
            reward = - self.sim_reject / 5
        isl_info = np.hstack(self.facility.isl_info)
        arrive_info = np.hstack(self.facility.arrive_info)
        dwell_info = np.hstack(self.facility.dwell_info)
        veh_info = np.array([current_event, current_dwell])
        veh_info = veh_info.reshape(2,1)
        # arrive_info = self.transformer.fit_transform(arrive_info)
        # dwell_info = self.transformer.fit_transform(dwell_info)
        # veh_info = self.transformer.fit_transform(veh_info)
        layout_info = np.stack([arrive_info, dwell_info], axis=0)
        veh_info = veh_info.reshape(2)
        state = [layout_info, veh_info]
        end_time = time.time()
        if end_time - start_time > 5:
            print("Encounter Infeasible Layout")
        return state, reward, end, None
