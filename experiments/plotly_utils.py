# plot the camera

import numpy as np
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
import plotly.express as px

def plot_camera(fig, rotation_matrix, translation_vector, focal_length, legend_name, show_legend):
    # Define the camera center
    center = translation_vector

    # Calculate the camera frame size based on the focal length
    frame_width = 2 * focal_length
    frame_height = 2 * focal_length

    # Calculate the corner points of the camera frame
    corner_points = [
        center + rotation_matrix @ np.array([frame_width / 2, frame_height / 2, focal_length]),
        center + rotation_matrix @ np.array([-frame_width / 2, frame_height / 2, focal_length]),
        center + rotation_matrix @ np.array([-frame_width / 2, -frame_height / 2, focal_length]),
        center + rotation_matrix @ np.array([frame_width / 2, -frame_height / 2, focal_length])
    ]

    # Add the camera frame as lines
    fig.add_trace(go.Scatter3d(
        x=[corner_points[0][0], corner_points[1][0], corner_points[2][0], corner_points[3][0], corner_points[0][0]],
        y=[corner_points[0][1], corner_points[1][1], corner_points[2][1], corner_points[3][1], corner_points[0][1]],
        z=[corner_points[0][2], corner_points[1][2], corner_points[2][2], corner_points[3][2], corner_points[0][2]],
        mode='lines',
        line=dict(color='blue', width=2),
        legendgroup=legend_name,
        name=legend_name,
        showlegend=show_legend,
        hoverinfo='text',
        text=legend_name
    ))

    # Connect the corners to the camera center
    for point in corner_points:
        fig.add_trace(go.Scatter3d(
            x=[point[0], center[0]],
            y=[point[1], center[1]],
            z=[point[2], center[2]],
            mode='lines',
            line=dict(color='blue', width=2),
            legendgroup=legend_name,
            name='',
            showlegend=False,
            hoverinfo='text',
            text=legend_name
        ))

def plot_point(fig, point, legend_name, color=None, text=''):
    fig.add_trace(go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode='markers',
        marker=dict(size=5, color=color),
        name=legend_name,
        hoverinfo='text',
        text=text if text else legend_name
    ))
    fig.update_layout(scene_aspectmode='cube')

def plot_points(fig, points, legend_name, color=None, text='', marker=None):
    if marker is None:
        marker = dict(size=5, color=color)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=marker,
        name=legend_name,
        hoverinfo='text',
        text=text if text else legend_name
    ))

def plot_ray_from_camera(fig, camera_pose, focal_length, point, legend_name):
    # Extract camera center and rotation matrix from the camera pose
    center = camera_pose[:3]
    rotation_matrix = camera_pose[3:]

    # Calculate the direction vector from the camera center to the point
    direction = point - center

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Calculate the end point of the ray using the focal length
    ray_end = center + direction * focal_length

    # Add the ray as a line segment
    fig.add_trace(go.Scatter3d(
        x=[center[0], ray_end[0]],
        y=[center[1], ray_end[1]],
        z=[center[2], ray_end[2]],
        mode='lines',
        line=dict(color='green', width=2),
        legendgroup=legend_name,
        name=legend_name
    ))

def plot_pc(fig, pc, legend_name, marker_line_width=0, marker_line_color='black', marker_size=5):
    points = pc.coords
    colors = pc.select_channels(['R', 'G', 'B', 'A']).clip(0, 1)
    colors[:,:3] = colors[:,:3] * 255.0
    colors = [f'rgba({r:.3f},{g:.3f},{b:.3f},{a:.3f})' for r,g,b,a in colors]
    text = [f'({x},{y},{z})' for x,y,z in points]
    marker = dict(size=marker_size, color=colors, line=dict(width=marker_line_width, color=marker_line_color))
    plot_points(fig, points, legend_name, colors, text, marker)

def show_image(img, hoverinfo='x+y+z', name=''):
    fig = px.imshow(img)
    fig.update_traces(hoverinfo=hoverinfo, name=name)
    fig.show()

def show_depth(depth, max_depth=30, hoverinfo='x+y+z', name=''):
    clipped_depth = depth.clip(0, max_depth)
    fig = px.imshow(clipped_depth, color_continuous_scale='gray')
    fig.update_traces(hoverinfo=hoverinfo, name=name)
    fig.show()
