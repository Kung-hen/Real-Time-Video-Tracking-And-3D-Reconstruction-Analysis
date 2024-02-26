import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import simpledialog

def show_animat(name):
    filename_prefix = name
    point_data = np.load(
        f"result/results_3d/{filename_prefix}/{filename_prefix}_point3d_trans.npy")
    chessboard = np.load(
        f"result/results_3d/{filename_prefix}/{filename_prefix}_chess_corner_trans.npy")

    corr_points = [40, 41, 32]
    X, Y, Z = [], [], []

    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(projection='3d')

    def update(t):
        ax.cla()
        for j in range(len(chessboard)):
            x_cb = chessboard[j][0]
            y_cb = chessboard[j][1]
            z_cb = chessboard[j][2]
            if j in corr_points:
                ax.scatter(x_cb, y_cb, z_cb, s=60, marker="*")
            else:
                ax.scatter(x_cb, y_cb, z_cb)
        for i in range(len(point_data)):
            x = point_data[i][t][0]
            y = point_data[i][t][1]
            z = point_data[i][t][2]
            X.append(x)
            Y.append(y)
            Z.append(z)
            ax.scatter(x, y, z, s=80, marker='o')
            ax.text(x, y, z, f"  3D Point{i+1}", size=8, zorder=3)
        ax.plot(X, Y, Z, 'green', alpha=0.5)
        ax.plot([0, 0], [0, 0], [0, 300], color="green")
        ax.plot([0, 0], [0, 320], [0, 0], color="black")
        ax.plot([0, 350], [0, 0], [0, 0], color="gray")
        ax.set_xlabel('$Coordinate-X$', fontsize=20)
        ax.set_ylabel('$Coordinate-Y$', fontsize=20)
        ax.set_zlabel('$Coordinate-Z$', fontsize=20)

    ani = FuncAnimation(fig=fig, func=update, frames=len(
        point_data[0]), interval=100, repeat=False)

    plt.show()

root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 弹出窗口提示用户输入数字
user_input = simpledialog.askinteger("輸入病歷號", "請輸入病歷號:")

# 处理用户输入
if user_input is not None:
    print("您输入的数字是:", user_input)
else:
    print("用户取消了输入")

# 关闭窗口
root.destroy()
show_animat(user_input)
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.animation import FuncAnimation


# X = []
# Y = []
# Z = []


# def update(t):
#     ax.cla()

#     x = np.cos(t/10)
#     print(x)
#     y = np.sin(t/10)
#     z = t/10

#     X.append(x)
#     Y.append(y)
#     Z.append(z)

#     ax.scatter(x, y, z, s=100, marker='o')
#     ax.plot(X, Y, Z)

#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.set_zlim(-1, 10)


# fig = plt.figure(dpi=100)
# ax = fig.add_subplot(projection='3d')

# ani = FuncAnimation(fig=fig, func=update, frames=100,
#                     interval=100, repeat=False)

plt.show()
