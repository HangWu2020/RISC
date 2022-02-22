import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)

def rot_points(points):
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    points = np.matmul(R,points.transpose()).transpose()
    return points    
    

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_single_pcd(points, view_angle=0.0, elev=15.0, show_axis=False, save_fig=False, save_path='point_cloud.png', point_size=0.5, cmap='brg'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=view_angle)
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    #rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    #pcd = pcd.transform(rotation_matrix)
    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap=cmap, marker='o', s=point_size, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    if not show_axis:
        plt.axis('off')
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    if save_fig:
        plt.savefig(save_path, format='png', dpi=800)
    plt.show()

    
if __name__ == '__main__':
    import h5py
    from hyper import *
    
    input_file = h5py.File('../d02_data/dataset/mvp_train_input.h5', 'r')
    input_data = np.array((input_file['incomplete_pcds'][()]))
    labels = np.array((input_file['labels'][()]))
    print (np.shape(labels)[0]/26)
    
    pc_id = 400*26
    points = rot_points(input_data[pc_id])
    
    plot_single_pcd(points, view_angle=30.0, show_axis=False, save_fig=False, save_path='point_cloud.png')
