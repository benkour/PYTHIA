
import os
from pathlib import Path
import sys
import numpy as np
import random
import torch
from helpers import *
from scipy.spatial.distance import cdist
import json

from utils.helpers import read_last_snapshot

sys.path.append(str(Path(__file__).resolve().parent.parent / 'bo_pipeline'))
current_script_path = Path(__file__).resolve()
with open(os.environ.get('BO_CONFIG', current_script_path.parent.parent / 'configs' / 'config_bo.json'), 'r') as json_file:
    config = json.load(json_file)


class Adsorbate_Molecule:
    def __init__(self, coordinates, theta_x, theta_y, theta_z):
        self.coordinates = np.array(coordinates)
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z
        self.center_of_mass = self.calculate_center_of_mass()
        self.rotated_coordinates = self.rotate_molecule()

    def calculate_center_of_mass(self):
        return np.mean(self.coordinates, axis=0)

    def rotation_matrix_x(self, theta):
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])

    def rotation_matrix_y(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def rotation_matrix_z(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    def rotate_molecule(self):
        R_x = self.rotation_matrix_x(self.theta_x)
        R_y = self.rotation_matrix_y(self.theta_y)
        R_z = self.rotation_matrix_z(self.theta_z)

        rotated_coords = [
            np.dot(R_z, np.dot(R_y, np.dot(R_x, coord -
                   self.center_of_mass))) + self.center_of_mass
            for coord in self.coordinates
        ]
        return np.array(rotated_coords)


def place_adsorbate(atoms, surface_coord, adsorbate_coordinates, min_distances, min_dist):


    min_dist = np.zeros(
        (adsorbate_coordinates.shape[0], len(atoms)-adsorbate_coordinates.shape[0]))

    for i in range(len(atoms) - adsorbate_coordinates.shape[0]):
        atom = atoms[i]
        for j in range(adsorbate_coordinates.shape[0]):
            min_dist[j, i] = min_distances[atoms[len(
                atoms) - adsorbate_coordinates.shape[0] + j]][atom]




    # read placement parameters from the config file (BO_CONFIG overrides config.json)


    adsorbate_xyz_path = config.get('adsorbate_xyz', 'adsorbate.xyz')
    adsorbate_xyz_path = os.path.join(current_script_path.parent,  adsorbate_xyz_path)
    adsorbate_coordinates = read_last_snapshot(adsorbate_xyz_path)
    # Extract the NR_OF_ABSORBANTS value from the configuration
    NR_OF_ABSORBANTS = adsorbate_coordinates.shape[0]
    while True:
        # Choose a random atom from the surface
        random_index = random.randint(
            0, len(atoms) - adsorbate_coordinates.shape[0] - 1)
        surface_atom = atoms[random_index]
        if surface_atom in config['non_viable_surface_atoms']:
            continue
        random_surface_atom = random.choice(surface_coord)
        if config['specific_atom_closest']:
            radius = min_distances[atoms[len(
                atoms) + config["which_atom"]]][surface_atom]
        else:
            radius = min_distances[atoms[len(
                atoms) - NR_OF_ABSORBANTS + random.choice(np.arange(NR_OF_ABSORBANTS))]][surface_atom]
        # Generate random spherical coordinates for the reference adsorbate atom
        radius = radius
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, 2 * np.pi)

        # Convert spherical coordinates to Cartesian coordinates
        reference_adsorbate_atom = (
            random_surface_atom[0] + radius * np.sin(phi) * np.cos(theta),
            random_surface_atom[1] + radius * np.sin(phi) * np.sin(theta),
            random_surface_atom[2] + radius * np.cos(phi)
        )

        # Choose a random atom from the adsorbate molecule as the reference
        # random_adsorbate_atom = random.choice(adsorbate_coordinates)
        if config['specific_atom_closest']:
            random_adsorbate_atom = adsorbate_coordinates[config['which_atom']]
        else:
            random_adsorbate_atom = random.choice(adsorbate_coordinates)
        # Calculate the translation vector
        translation_vector = np.array(
            reference_adsorbate_atom) - np.array(random_adsorbate_atom)

        # Translate the adsorbate coordinates to place the reference atom correctly
        translated_adsorbate_coords = [
            np.array(coord) + translation_vector for coord in adsorbate_coordinates
        ]

        theta_x = random.uniform(0, 2 * np.pi)
        theta_y = random.uniform(0, 2 * np.pi)
        theta_z = random.uniform(0, 2 * np.pi)

        translated_adsorbate_coords = Adsorbate_Molecule(
            translated_adsorbate_coords, theta_x, theta_y, theta_z).rotated_coordinates
        # get the pariwise distances of every surface atom to every adsorbate atom

        # print("distance shape", distances.shape)

        periodicity = config['periodicity']
        if periodicity:
            system_coords = np.concatenate(
                (surface_coord, translated_adsorbate_coords), axis=0)
            pairwise_distances = get_periodic_distance(
                system_coords, config)

            pairwise_distances = pairwise_distances[-NR_OF_ABSORBANTS:,
                                                    :-NR_OF_ABSORBANTS]
        else:
            pairwise_distances = cdist(
                translated_adsorbate_coords, surface_coord)

        if config['specific_atom_closest']:
            if config['probability_of_closest'] >= np.random.rand():
                specific_atom_distance = np.min(
                    pairwise_distances[config['which_atom'], :])
                if specific_atom_distance != np.min(pairwise_distances):
                    continue
        # Check if some adsorbate atoms are above and some are below the surface
        surface_cluster = config['surface_cluster']
        if surface_cluster == "surface":
            if periodicity:

                surface_z = surface_coord[:, -1]
                adsorbate_z = translated_adsorbate_coords[:, -1]

                # Create a matrix of distances
                distance_matrix = surface_z[:,
                                            np.newaxis] - adsorbate_z[np.newaxis, :]

                # Transpose the matrix to get the desired shape
                distance_matrix = distance_matrix.T
                is_feasible = not (np.any(distance_matrix > 0)
                                   and np.any(distance_matrix < 0))
                # check if the adsorbate is outside the x and y bounds of the surface
                surface_x_bounds = (np.min(surface_coord[:, 0]), np.max(surface_coord[:, 0]))
                surface_y_bounds = (np.min(surface_coord[:, 1]), np.max(surface_coord[:, 1]))
                adsorbate_x_bounds = (np.min(translated_adsorbate_coords[:, 0]), np.max(translated_adsorbate_coords[:, 0]))
                adsorbate_y_bounds = (np.min(translated_adsorbate_coords[:, 1]), np.max(translated_adsorbate_coords[:, 1]))
                if (adsorbate_x_bounds[0] < surface_x_bounds[0] or adsorbate_x_bounds[1] > surface_x_bounds[1] or
                        adsorbate_y_bounds[0] < surface_y_bounds[0] or adsorbate_y_bounds[1] > surface_y_bounds[1]):
                    is_feasible = False
            else:
                closest_surface_atom = np.argmin(pairwise_distances, axis=1)
                closest_surface_coords = surface_coord[closest_surface_atom]
                z_sign = closest_surface_coords[:, -1] - \
                    translated_adsorbate_coords[:, -1]
                is_feasible = not (np.any(z_sign > 0) and np.any(z_sign < 0))

        elif surface_cluster == "cluster":
            is_feasible = cluster_convex_hull(
                translated_adsorbate_coords, surface_coord)
        if is_feasible:

            if np.min(pairwise_distances - min_dist) > 0:
                return torch.tensor(translated_adsorbate_coords)
        else:
            continue