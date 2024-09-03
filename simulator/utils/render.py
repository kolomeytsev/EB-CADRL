import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import numpy as np
from numpy.linalg import norm


plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

X_OFFSET = 0.11
Y_OFFSET = 0.11
CMAP = plt.cm.get_cmap("hsv", 10)
# ROBOT_COLOR = (1, 0.34, 0.114)
ROBOT_COLOR = "orange"
GOAL_COLOR = "red"
ARROW_COLOR = "red"
ARROW_STYLE = patches.ArrowStyle("->", head_length=8, head_width=8)


def render_trajectory(
    states,
    adults_to_render,
    bicycles_to_render,
    children_to_render,
    robot_radius,
    time_step,
    obstacle_vertices,
    last_circle_radius,
):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.tick_params(labelsize=20)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("x(m)", fontsize=22)
    ax.set_ylabel("y(m)", fontsize=22)

    robot_positions = [states[i][0].position for i in range(len(states))]

    adult_positions = [
        [state[1][j].position for j in range(len(adults_to_render))] for state in states
    ]

    bicycle_positions = [
        [state[2][j].position for j in range(len(bicycles_to_render))]
        for state in states
    ]

    children_positions = [
        [state[2][j].position for j in range(len(children_to_render))]
        for state in states
    ]

    for k in range(len(states)):
        if k % 4 == 0 or k == len(states) - 1:
            robot = plt.Circle(
                robot_positions[k], robot_radius, fill=True, color=ROBOT_COLOR
            )

            adults = [
                plt.Circle(
                    adult_positions[k][i],
                    adults_to_render[i].radius,
                    fill=False,
                    color=CMAP(a),
                )
                for i, a in zip(
                    range(len(adults_to_render)), range(len(adults_to_render))
                )
            ]
            ax.add_artist(robot)
            for adult in adults:
                ax.add_artist(adult)

            bicycles = [
                plt.Circle(
                    bicycle_positions[k][i],
                    bicycles_to_render[i].radius,
                    fill=False,
                    color=CMAP(a),
                )
                for i, a in zip(
                    range(len(bicycles_to_render)), range(len(bicycles_to_render))
                )
            ]
            for bicycle in bicycles:
                ax.add_artist(bicycle)

            children = [
                plt.Circle(
                    children_positions[k][i],
                    children_to_render[i].radius,
                    fill=False,
                    color=CMAP(a),
                )
                for i, a in zip(
                    range(len(children_to_render)), range(len(children_to_render))
                )
            ]
            for child in children:
                ax.add_artist(child)

        # add time annotation
        global_time = k * time_step
        if global_time % 4 == 0 or k == len(states) - 1:
            agents = adults + [robot]
            times = list()
            for i in range(len(agents)):
                if (
                    global_time > 0
                    and norm(
                        [
                            prev_loc[i][0] - agents[i].center[0],
                            prev_loc[i][1] - agents[i].center[1],
                        ]
                    )
                    < 0.2
                ):
                    continue
                else:
                    times.append(
                        plt.text(
                            agents[i].center[0] - X_OFFSET,
                            agents[i].center[1] - Y_OFFSET,
                            "{:.1f}".format(global_time),
                            color="black",
                            fontsize=16,
                        )
                    )

            prev_loc = [
                [agents[i].center[0] - X_OFFSET, agents[i].center[1]]
                for i in range(len(agents))
            ]
            for time in times:
                ax.add_artist(time)
        if k != 0:
            nav_direction = plt.Line2D(
                (states[k - 1][0].px, states[k][0].px),
                (states[k - 1][0].py, states[k][0].py),
                color=ROBOT_COLOR,
                ls="solid",
            )
            adult_directions = [
                plt.Line2D(
                    (states[k - 1][1][i].px, states[k][1][i].px),
                    (states[k - 1][1][i].py, states[k][1][i].py),
                    color=CMAP(i),
                    ls="solid",
                )
                for i in range(len(adults_to_render))
            ]
            ax.add_artist(nav_direction)
            for adult_direction in adult_directions:
                ax.add_artist(adult_direction)
    obstacles = [plt.Polygon(obstacle_vertex) for obstacle_vertex in obstacle_vertices]

    for obstacle in obstacles:
        ax.add_artist(obstacle)
    goal = lines.Line2D(
        [0],
        [last_circle_radius],
        color=GOAL_COLOR,
        marker="*",
        linestyle="None",
        markersize=40,
        label="Goal",
    )
    ax.add_artist(goal)
    fig2 = plt.figure()
    plt.axis("off")
    plt.legend(
        [goal] + [robot] + [adult for adult in adults],
        ["Goal"] + ["Robot"] + ["Adult " + str(i) for i in range(len(adults))],
        fontsize=22,
        loc=3,
    )
    plt.show()


def render_am(
    frame,
    states,
    obstacle_vertices,
    local_maps_angular,
    angular_map_max_range,
    angular_map_dim,
    angular_map_max_angle,
    angular_map_min_angle,
    robot_radius,
):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.tick_params(labelsize=22)
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_xlabel("x(m)", fontsize=22)
    ax.set_ylabel("y(m)", fontsize=22)
    robot_position = states[frame][0].position
    robot = plt.Circle(robot_position, robot_radius, fill=True, color=ROBOT_COLOR)
    ax.add_artist(robot)
    obstacles = [plt.Polygon(obstacle_vertex) for obstacle_vertex in obstacle_vertices]
    for obstacle in obstacles:
        ax.add_artist(obstacle)
    orientation = [
        (states[frame][0].px, states[frame][0].py),
        (
            states[frame][0].px + robot_radius * np.cos(states[frame][0].theta),
            states[frame][0].py + robot_radius * np.sin(states[frame][0].theta),
        ),
    ]
    arrows = [
        patches.FancyArrowPatch(*orientation, color=ARROW_COLOR, arrowstyle=ARROW_STYLE)
    ]
    for arrow in arrows:
        ax.add_artist(arrow)

    # Plot angular map
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.tick_params(labelsize=22)
    ax2.set_xlim(-7.5, 7.5)
    ax2.set_ylim(-7.5, 7.5)
    ax2.set_xlabel("x(m)", fontsize=22)
    ax2.set_ylabel("y(m)", fontsize=22)
    angular_resolution = (angular_map_max_angle - angular_map_min_angle) / float(
        angular_map_dim
    )

    CMAP = plt.get_cmap("gnuplot")

    for i in range(angular_map_dim):
        angle_start = (
            angular_map_min_angle + i * angular_resolution
        ) * 180 / np.pi + 90
        angle_end = (
            angular_map_min_angle + (i + 1) * angular_resolution
        ) * 180 / np.pi + 90

        distance_cone = plt.matplotlib.patches.Wedge(
            (0.0, 0.0),
            local_maps_angular[frame][i] * angular_map_max_range,
            angle_start,
            angle_end,
            facecolor=CMAP(local_maps_angular[frame][i]),
            alpha=0.5,
        )
        ax2.add_artist(distance_cone)

    plt.show()


def render_traj_3D(states, last_circle_radius, adults_to_render):
    """
    Ox: Meters
    Oy: Meters
    Oz: Timestamp
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.tick_params(labelsize=20)
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-5.4, 5.4)
    ax.set_xlabel("x(m)", fontsize=22)
    ax.set_ylabel("y(m)", fontsize=22)
    ax.set_zlabel("Timestep", fontsize=22)

    robot_positions = [states[i][0].position for i in range(len(states))]
    adult_positions = [
        [states[i][1][j].position for j in range(len(adults_to_render))]
        for i in range(len(states))
    ]
    for k in range(len(states)):
        if k % 4 == 0 or k == len(states) - 1:
            robot = (robot_positions[k][0], robot_positions[k][1], k)
            adults = [
                (adult_positions[k][i][0], adult_positions[k][i][1], k)
                for i, a in zip(
                    range(len(adults_to_render)), range(len(adults_to_render))
                )
            ]
            ax.scatter(robot[0], robot[1], robot[2], c=CMAP(0))
            for i, adult in enumerate(adults):
                ax.scatter(adult[0], adult[1], adult[2], c=CMAP(i + 1))
    goal = lines.Line2D(
        [0],
        [last_circle_radius],
        color=GOAL_COLOR,
        marker="*",
        linestyle="None",
        markersize=40,
        label="Goal",
    )
    ax.add_artist(goal)
    plt.show()


def render_og(
    frame,
    states,
    obstacle_vertices,
    use_grid_map,
    local_maps,
    submap_size_m,
    map_resolution,
    robot_radius,
):
    if not use_grid_map:
        print("use_grid_map is false, cannot render grid")
        return

    assert len(local_maps) > 0, "local_maps must not be empty"
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.tick_params(labelsize=22)
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-7.5, 7.5)
    ax.set_xlabel("x(m)", fontsize=22)
    ax.set_ylabel("y(m)", fontsize=22)

    robot_position = states[frame][0].position
    robot = plt.Circle(robot_position, robot_radius, fill=True, color=ROBOT_COLOR)
    ax.add_artist(robot)
    obstacles = [plt.Polygon(obstacle_vertex) for obstacle_vertex in obstacle_vertices]
    for obstacle in obstacles:
        ax.add_artist(obstacle)
    orientation = [
        (states[frame][0].px, states[frame][0].py),
        (
            states[frame][0].px + robot_radius * np.cos(states[frame][0].theta),
            states[frame][0].py + robot_radius * np.sin(states[frame][0].theta),
        ),
    ]
    arrows = [
        patches.FancyArrowPatch(*orientation, color=ARROW_COLOR, arrowstyle=ARROW_STYLE)
    ]
    for arrow in arrows:
        ax.add_artist(arrow)
    # Plot occupancy grid
    fig2, ax2 = plt.subplots(figsize=(7.5, 7.5))
    ax2.tick_params(labelsize=22)
    img = Image.fromarray(local_maps[frame].astype("uint8"))
    img_rotated = img.rotate(90)
    ax2.tick_params(labelsize=16)
    ax2.imshow(img_rotated, cmap="gray", vmin=0, vmax=1)
    middle_idx = int(round(submap_size_m / 2.0 / map_resolution))
    robot2 = plt.Circle(
        (middle_idx, middle_idx),
        int(round(robot_radius / map_resolution)),
        fill=True,
        color=ROBOT_COLOR,
    )
    ax2.add_patch(robot2)

    plt.show()


def render_video(
    states,
    last_circle_radius,
    robot_radius,
    adults_to_render,
    bicycles_to_render,
    children_to_render,
    static_obstacles_as_pedestrians,
    local_maps,
    use_grid_map,
    obstacle_vertices,
    robot_kinematics,
    adult_num,
    submap_size_m,
    map_resolution,
    time_step,
    angular_map_max_angle,
    angular_map_min_angle,
    angular_map_dim,
    angular_map_max_range,
    local_maps_angular,
    other_robots_to_render,
    attention_weights,
    output_file=None,
    deconv=None,
    plot_agents_goals_flag=True,
):
    # fig, ax = plt.subplots(figsize=(20, 20))
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.tick_params(labelsize=20)
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-7.0, 7.0)
    ax.set_xlabel("x(m)", fontsize=22)
    ax.set_ylabel("y(m)", fontsize=22)

    # add robot and its goal

    robot_positions = [state[0].position for state in states]
    goal = lines.Line2D(
        [0],
        [last_circle_radius],
        color=GOAL_COLOR,
        marker="*",
        linestyle="None",
        markersize=40,
        label="Goal",
    )
    robot = plt.Circle(
        robot_positions[0],
        robot_radius,
        fill=True,
        edgecolor="k",
        facecolor=ROBOT_COLOR,
    )

    ax.add_artist(robot)
    ax.add_artist(goal)
    legends = [robot]

    def plot_agents_goals(ax, agents, color):
        for i, agent in enumerate(agents):
            agents_goal = lines.Line2D(
                [agent.gx],
                [agent.gy],
                color=color,
                marker="*",
                linestyle="None",
                markersize=30,
            )
            number = plt.text(
                agent.gx - X_OFFSET,
                agent.gy - Y_OFFSET,
                str(i),
                color="black",
                fontsize=22,
            )
            ax.add_artist(agents_goal)
            ax.add_artist(number)
        return agents_goal

    if plot_agents_goals_flag:
        plot_agents_goals(ax, adults_to_render, "tab:blue")
        plot_agents_goals(ax, bicycles_to_render, "tab:cyan")
        plot_agents_goals(ax, children_to_render, "tab:green")

    ADULT_STATE_INDEX = 1
    BICYCLE_STATE_INDEX = 2
    CHILD_STATE_INDEX = 3

    # adults
    adult_positions = [
        [state[ADULT_STATE_INDEX][j].position for j in range(len(adults_to_render))]
        for state in states
    ]
    adults = [
        plt.Circle(
            adult_positions[0][i],
            adults_to_render[i].radius,
            fill=True,
            edgecolor="k",
            facecolor="tab:blue",
        )
        for i in range(len(adults_to_render))
    ]
    legends.append(adults[0])
    adult_numbers = [
        plt.text(
            adults[i].center[0] - X_OFFSET,
            adults[i].center[1] - Y_OFFSET,
            str(i),
            color="black",
            fontsize=22,
        )
        for i in range(len(adults_to_render))
    ]

    for i, adult in enumerate(adults):
        ax.add_artist(adult)
        ax.add_artist(adult_numbers[i])

    # bicycles
    bicycle_positions = [
        [state[BICYCLE_STATE_INDEX][j].position for j in range(len(bicycles_to_render))]
        for state in states
    ]
    bicycles = [
        plt.Circle(
            bicycle_positions[0][i],
            bicycles_to_render[i].radius,
            fill=True,
            color="tab:cyan",
        )
        for i in range(len(bicycles_to_render))
    ]
    legends.append(bicycles[0])
    bicycle_numbers = [
        plt.text(
            bicycles[i].center[0] - X_OFFSET,
            bicycles[i].center[1] - Y_OFFSET,
            str(i),
            color="black",
            fontsize=22,
        )
        for i in range(len(bicycles_to_render))
    ]

    for i, bicycle in enumerate(bicycles):
        ax.add_artist(bicycle)
        ax.add_artist(bicycle_numbers[i])

    # children
    children_positions = [
        [state[CHILD_STATE_INDEX][j].position for j in range(len(children_to_render))]
        for state in states
    ]
    children = [
        plt.Circle(
            children_positions[0][i],
            children_to_render[i].radius,
            fill=True,
            edgecolor="k",
            facecolor="tab:green",
        )
        for i in range(len(children_to_render))
    ]
    legends.append(children[0])
    children_numbers = [
        plt.text(
            children[i].center[0] - X_OFFSET,
            children[i].center[1] - Y_OFFSET,
            str(i),
            color="black",
            fontsize=22,
        )
        for i in range(len(children_to_render))
    ]

    for i, child in enumerate(children):
        ax.add_artist(child)
        ax.add_artist(children_numbers[i])

    """
    other_robot_positions = [[state[3][j].position for j in range(
        len(other_robots))] for state in states]
    other_robots = [
        plt.Circle(
            other_robot_positions[0][i],
            other_robots[i].radius,
            fill=False,
            color=ROBOT_COLOR) for i in range(
            len(other_robots))]
    other_robot_numbers = [
        plt.text(
            other_robots[i].center[0] -
            X_OFFSET,
            other_robots[i].center[1] -
            Y_OFFSET,
            str(i),
            color='black',
            fontsize=12) for i in range(
            len(other_robots))]

    for i, other_robot in enumerate(other_robots):
        ax.add_artist(other_robot)
        ax.add_artist(other_robot_numbers[i])
    """

    # if static_obstacles_as_pedestrians is not None:
    #    for adult in static_obstacles_as_pedestrians:
    #        circle = plt.Circle(adult.position, adult.radius, color='r', fill=False)
    #        plt.gca().add_patch(circle)

    obstacles = [
        plt.Polygon(obstacle_vertex, color="tab:gray")
        for obstacle_vertex in obstacle_vertices
    ]

    for i, obstacle in enumerate(obstacles):
        ax.add_artist(obstacle)

    legends.append(obstacles[0])
    legends.append(goal)
    plt.legend(
        legends,
        ["Robot", "Adult", "Bicyclist", "Child", "Obstacle", "Robot's Goal"],
        fontsize=22,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        ncol=3,
    )

    # add time annotation
    time = plt.text(-1.5, 7.5, "Time: {}".format(0), fontsize=24)
    ax.add_artist(time)
    # compute orientation in each step and use arrow to show the
    # direction
    radius = robot_radius
    orientations_adults = [
        [
            (
                (state[1][j].px, state[1][j].py),
                (
                    state[1][j].px
                    + 1.5 * adults_to_render[j].radius * np.cos(state[1][j].theta),
                    state[1][j].py
                    + 1.5 * adults_to_render[j].radius * np.sin(state[1][j].theta),
                ),
            )
            for state in states
        ]
        for j in range(len(adults_to_render))
    ]
    """orientations_other_robots = [
        [
            ((state[2][j].px,
                state[2][j].py),
                (state[2][j].px +
                    1.5 *
                    other_robots_to_render[j].radius *
                    np.cos(
                    state[2][j].theta),
                    state[2][j].py +
                    1.5 *
                    other_robots_to_render[j].radius *
                    np.sin(
                    state[2][j].theta))) for state in states] for j in range(
            len(
                other_robots_to_render))]"""

    """
    orientation_self = [
        ((state[0].px,
            state[0].py),
            (state[0].px +
                radius *
                np.cos(
                state[0].theta),
                state[0].py +
                radius *
                np.sin(
                state[0].theta))) for state in states]

    arrow_self = patches.FancyArrowPatch(
        *orientation_self[0], color=ARROW_COLOR, arrowstyle=ARROW_STYLE)
    ax.add_artist(arrow_self)
    orientations = orientations_adults
    # orientations.extend(orientations_other_robots)
    arrows_others = [
        patches.FancyArrowPatch(
            *orientation[0],
            color='red',
            arrowstyle=ARROW_STYLE) for orientation in orientations]
    for arrow in arrows_others:
        ax.add_artist(arrow)
    """

    radius = robot_radius
    if robot_kinematics == "unicycle":
        orientation = [
            (
                (state[0].px, state[0].py),
                (
                    state[0].px + radius * np.cos(state[0].theta),
                    state[0].py + radius * np.sin(state[0].theta),
                ),
            )
            for state in states
        ]
        orientations = [orientation]
    else:
        orientations = []
        for i in range(adult_num + 1):
            orientation = []
            for state in states:
                if i == 0:
                    agent_state = state[0]
                else:
                    agent_state = state[1][i - 1]
                theta = np.arctan2(agent_state.vy, agent_state.vx)
                orientation.append(
                    (
                        (agent_state.px, agent_state.py),
                        (
                            agent_state.px + radius * np.cos(theta),
                            agent_state.py + radius * np.sin(theta),
                        ),
                    )
                )
            orientations.append(orientation)

    arrows = [
        patches.FancyArrowPatch(
            *orientation[0], color=ARROW_COLOR, arrowstyle=ARROW_STYLE
        )
        for orientation in orientations
    ]
    for arrow in arrows:
        ax.add_artist(arrow)
    global_step = 0

    if use_grid_map:
        # Plot robot's view
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        img = Image.fromarray(local_maps[0].astype("uint8"))
        img_rotated = img.rotate(90)
        ax2.tick_params(labelsize=16)
        ax2.imshow(img_rotated, cmap="gray", vmin=0, vmax=1)
        middle_idx = int(round(submap_size_m / 2.0 / map_resolution))
        robot2 = plt.Circle(
            (middle_idx, middle_idx),
            int(round(robot_radius / map_resolution)),
            fill=True,
            color="red",
        )
        ax2.add_patch(robot2)
        time2 = plt.text(5, 5, "Time: {}".format(0), fontsize=22)
        # ax2.add_artist(time2)
    else:
        # Plot robot's view
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.tick_params(labelsize=16)
        ax2.set_xlim(-7, 7)
        ax2.set_ylim(-7, 7)
        angular_resolution = (angular_map_max_angle - angular_map_min_angle) / float(
            angular_map_dim
        )

        CMAP = plt.get_cmap("gnuplot")

        for ii in range(angular_map_dim):
            angle_start = (
                angular_map_min_angle + ii * angular_resolution
            ) * 180 / np.pi + 90
            angle_end = (
                angular_map_min_angle + (ii + 1) * angular_resolution
            ) * 180 / np.pi + 90

            distance_cone = plt.matplotlib.patches.Wedge(
                (0.0, 0.0),
                local_maps_angular[0][ii] * angular_map_max_range,
                angle_start,
                angle_end,
                facecolor=CMAP(local_maps_angular[0][ii]),
                alpha=0.5,
            )
            ax2.add_artist(distance_cone)
        time2 = plt.text(6, 8, "Time: {}".format(0), fontsize=22)
        ax2.add_artist(time2)

    if deconv is not None:
        fig3, ax3 = plt.subplots(figsize=(10, 10))
        grid_binary = np.zeros_like(deconv[0])
        THRESHOLD_VALUE = 0.9
        indeces = deconv[0] > THRESHOLD_VALUE
        grid_binary[indeces] = 1
        img = Image.fromarray(grid_binary.astype("uint8"))
        img_rotated = img.rotate(90)
        ax3.imshow(img_rotated, CMAP="gray", vmin=0, vmax=1)
        time3 = plt.text(5, 5, "Time: {}".format(0), fontsize=16)
        ax3.add_artist(time3)

    def update_env(frame_num):
        # nonlocal arrows_others
        # nonlocal arrow_self
        nonlocal arrows

        global_step = frame_num
        robot.center = robot_positions[frame_num]
        for i, adult in enumerate(adults):
            adult.center = adult_positions[frame_num][i]
            adult_numbers[i].set_position(
                (adult.center[0] - X_OFFSET, adult.center[1] - Y_OFFSET)
            )
            # if attention_weights is not None:
            #    adult.set_color(str(attention_weights[frame_num][i]))
            #    attention_scores[i].set_text('adult {}: {:.2f}'.format(i, attention_weights[frame_num][i]))

        for i, bicycle in enumerate(bicycles):
            bicycle.center = bicycle_positions[frame_num][i]
            bicycle_numbers[i].set_position(
                (bicycle.center[0] - X_OFFSET, bicycle.center[1] - Y_OFFSET)
            )

        for i, child in enumerate(children):
            child.center = children_positions[frame_num][i]
            children_numbers[i].set_position(
                (child.center[0] - X_OFFSET, child.center[1] - Y_OFFSET)
            )

        for arrow in arrows:
            arrow.remove()
        arrows = [
            patches.FancyArrowPatch(
                *orientation[frame_num], color=ARROW_COLOR, arrowstyle=ARROW_STYLE
            )
            for orientation in orientations
        ]
        for arrow in arrows:
            ax.add_artist(arrow)

        """arrow_self.remove()
        for arrow in arrows_others:
            arrow.remove()
        arrow_self = patches.FancyArrowPatch(
            *orientation_self[frame_num], color='black', arrowstyle=ARROW_STYLE)
        arrows_others = [
            patches.FancyArrowPatch(
                *orientation[frame_num],
                color='red',
                arrowstyle=ARROW_STYLE) for orientation in orientations]
        ax.add_artist(arrow_self)
        for arrow in arrows_others:
            ax.add_artist(arrow)"""

        # for i, other_robot in enumerate(other_robots):
        #     other_robot.center = other_robot_positions[frame_num][i]
        #     other_robot_numbers[i].set_position(
        #         (other_robot.center[0] - X_OFFSET, other_robot.center[1] - Y_OFFSET))

        time.set_text("Time: {:.2f}".format(frame_num * time_step))

    def update_static_map(frame_num):
        ax2.clear()
        if use_grid_map:
            img = Image.fromarray(local_maps[frame_num].astype("uint8"))
            img_rotated = img.rotate(90)
            ax2.imshow(img_rotated, CMAP="gray", vmin=0, vmax=1)
            middle_idx = int(round(submap_size_m / 2.0 / map_resolution))
            robot = plt.Circle(
                (middle_idx, middle_idx),
                int(round(robot_radius / map_resolution)),
                fill=True,
                color="red",
            )
            ax2.add_patch(robot)
        else:
            ax2.set_xlim(-7, 7)
            ax2.set_ylim(-7, 7)
            angular_resolution = (
                angular_map_max_angle - angular_map_min_angle
            ) / float(angular_map_dim)

            CMAP = plt.get_cmap("gnuplot")

            for ii in range(angular_map_dim):
                angle_start = (
                    angular_map_min_angle + ii * angular_resolution
                ) * 180 / np.pi + 90
                angle_end = (
                    angular_map_min_angle + (ii + 1) * angular_resolution
                ) * 180 / np.pi + 90
                distance_cone = plt.matplotlib.patches.Wedge(
                    (0.0, 0.0),
                    local_maps_angular[frame_num][ii] * angular_map_max_range,
                    angle_start,
                    angle_end,
                    facecolor=CMAP(local_maps_angular[frame_num][ii]),
                    alpha=0.5,
                )
                ax2.add_artist(distance_cone)
        time2.set_text("Time: {:.2f}".format(frame_num * time_step))
        # ax2.add_artist(time2)

    def update_dec(frame_num):
        ax3.clear()
        grid_binary = np.zeros_like(deconv[frame_num])
        THRESHOLD_VALUE = 0.7
        indeces = deconv[frame_num] > THRESHOLD_VALUE
        grid_binary[indeces] = 1
        img = Image.fromarray(grid_binary.astype("uint8"))
        img_rotated = img.rotate(90)
        ax3.imshow(img_rotated, CMAP="gray", vmin=0, vmax=1)
        time3.set_text("Time: {:.2f}".format(frame_num * time_step))
        ax3.add_artist(time3)

    anim = animation.FuncAnimation(
        fig, update_env, frames=len(states), interval=time_step * 1000
    )
    anim.running = True
    anim2 = animation.FuncAnimation(
        fig2, update_static_map, frames=len(states), interval=time_step * 1000
    )
    anim2.running = True
    if deconv is not None:
        anim3 = animation.FuncAnimation(
            fig3, update_dec, frames=len(states), interval=time_step * 1000
        )
        anim3.running = True

    plt.tight_layout(pad=1.6)
    if output_file is not None:
        ffmpeg_writer = animation.writers["ffmpeg"]
        writer = ffmpeg_writer(fps=8, metadata=dict(artist="Me"), bitrate=7200)
        anim.save(output_file, writer=writer)
        # anim2.save(output_file[:-4] + '_map.mp4', writer=writer)
    else:
        plt.show()
