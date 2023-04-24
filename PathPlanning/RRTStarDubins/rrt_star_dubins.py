"""
Path planning Sample Code with RRT and Dubins path

author: AtsushiSakai(@Atsushi_twi)

"""
import copy
import math
import random
from tracemalloc import start
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))  # root dir
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from DubinsPath import dubins_path_planner
from RRTStar.rrt_star import RRTStar
from utils.plot import plot_arrow

show_animation = True


class RRTStarDubins(RRTStar):
    """
    Class for RRT star planning with Dubins path
    """

    class Node(RRTStar.Node):
        """
        RRT Node
        """

        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 goal_sample_rate=10,
                 max_iter=2500,
                 connect_circle_dist=50.0,
                 robot_radius=0.0,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        robot_radius: robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist

        self.curvature = 1.0  # for dubins path
        self.goal_yaw_th = np.deg2rad(1.0)
        self.goal_xy_th = 0.5
        self.robot_radius = robot_radius

    def planning(self, animation=True, search_until_max_iter=True, global_path = None, starts = None):
        """
        RRT Star planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        #plt.plot(self.start.x, self.start.y, "xb") didn't do anything
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if animation and i % 5 == 0:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd, global_path, starts)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def draw_graph(self, rnd=None, global_path = None, starts = None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.gca().add_patch(Rectangle((ox - size / 2, oy - size / 2), size, size, facecolor = 'grey'))

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        if starts is not None:
            for i in range(len(starts)):
                p1, p2, p3 = starts[i]
                plt.plot(p1, p2, "xb")
        if global_path is not None:
            plt.plot([x for (x, y) in global_path], [y for (x, y) in global_path], '-r')
        plt.axis([0, 55, 0, 55])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = \
            dubins_path_planner.plan_dubins_path(
                from_node.x, from_node.y, from_node.yaw,
                to_node.x, to_node.y, to_node.yaw, self.curvature)

        if len(px) <= 1:  # cannot find a dubins path
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_lengths = dubins_path_planner.plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        cost = sum([abs(c) for c in course_lengths])

        return from_node.cost + cost

    def get_random_node(self):

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand),
                            random.uniform(-math.pi, math.pi)
                            )
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)

        return rnd

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        print("final")
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path


def main():
    print("Start rrt star with dubins planning")


    # ====Search Path with RRT====
    obstacleList = [
        (7.5, 2.5, 5),
        (7.5, 7.5, 5),
        (7.5, 17.5, 5),
        (7.5, 22.5, 5),
        (7.5, 32.5, 5),
        (7.5, 37.5, 5),
        (7.5, 47.5, 5),
        (7.5, 52.5, 5),
        (17.5, 2.5, 5),
        (17.5, 7.5, 5),
        (17.5, 17.5, 5),
        (17.5, 22.5, 5),
        (17.5, 32.5, 5),
        (17.5, 37.5, 5),
        (17.5, 47.5, 5),
        (17.5, 52.5, 5),
        (27.5, 2.5, 5),
        (27.5, 7.5, 5),
        (27.5, 17.5, 5),
        (27.5, 22.5, 5),
        (27.5, 32.5, 5),
        (27.5, 37.5, 5),
        (27.5, 47.5, 5),
        (27.5, 52.5, 5),
        (37.5, 2.5, 5),
        (37.5, 7.5, 5),
        (37.5, 17.5, 5),
        (37.5, 22.5, 5),
        (37.5, 32.5, 5),
        (37.5, 37.5, 5),
        (37.5, 47.5, 5),
        (37.5, 52.5, 5),
        (47.5, 2.5, 5),
        (47.5, 7.5, 5),
        (47.5, 17.5, 5),
        (47.5, 22.5, 5),
        (47.5, 32.5, 5),
        (47.5, 37.5, 5),
        (47.5, 47.5, 5),
        (47.5, 52.5, 5),
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    start = [2.5, 2.5, np.deg2rad(90)]
    goal = [42.5, 50, np.deg2rad(90)]

    

    rrtstar_dubins = RRTStarDubins(start, goal, rand_area=[0, 55], obstacle_list=obstacleList)
    path = rrtstar_dubins.planning(animation=show_animation)
    starts = [(2.5,2.5,np.deg2rad(90))]
    
    #keep last 10 for tcom of path
    size = len(path)
    x1, y1 = path[size-249]
    x2, y2 = path[size-251]
    dx = x2 - x1
    dy = y2 - y1
    ang = np.arctan2(dy, dx)
    iter_path = path[-250:]
    x, y = iter_path[0]
    new_start = (x,y,ang)
    starts.append(new_start) 
    global_path = list(tuple())
    local_path = list(tuple())
    for i in range(len(iter_path)):
        x, y = iter_path[i]
        local_path.append((x,y)) #add to beginning
    for i in range(len(local_path)):
        val = i+1
        x, y = local_path[-val:][0]
        global_path.insert(0,(x,y))

    while new_start != goal:
        rrtstar_dubins = RRTStarDubins(new_start, goal, rand_area=[0, 55], obstacle_list=obstacleList)
        path = rrtstar_dubins.planning(animation=show_animation, global_path = global_path, starts = starts)
        size = len(path)
        if size < 250:
            iter_path = path
            x, y = iter_path[0]
            new_start = (x,y,np.deg2rad(90))
            starts.append(new_start)
            local_path = list(tuple())
            for i in range(len(iter_path)):
                x, y = iter_path[i]
                local_path.append((x,y))
            for i in range(len(local_path)):
                val = i+1
                x, y = local_path[-val:][0]
                global_path.insert(0,(x,y))
            break
        else:
            x1, y1 = path[size-249]
            x2, y2 = path[size-251]
            dx = x2 - x1
            dy = y2 - y1
            ang = np.arctan2(dy, dx)
            iter_path = path[-250:]
            x, y = iter_path[0]
            new_start = (x,y,ang)
            starts.append(new_start)
            """
            for i in range(len(iter_path)):
                x, y = iter_path[-i:]
                global_path.insert(0,(x,y))
            """
            local_path = list(tuple())
            for i in range(len(iter_path)):
                x, y = iter_path[i]
                local_path.append((x,y))
            for i in range(len(local_path)):
                val = i+1
                x, y = local_path[-val:][0]
                global_path.insert(0,(x,y))

    # Draw final path
    if show_animation:  # pragma: no cover
        rrtstar_dubins.draw_graph(None, global_path, starts)
        plt.plot([x for (x, y) in global_path], [y for (x, y) in global_path], '-r')
        for i in range(len(starts)):
            p1, p2, p3 = starts[i]
            plt.plot(p1, p2, "xb")
        plt.grid(True)
        plt.pause(0.001)

        plt.show()


if __name__ == '__main__':
    main()
