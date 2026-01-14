"""
Clustering Tool for Exercise 1
Clustering Methods Course - University of Eastern Finland

This tool implements:
1. Data reading from http://cs.uef.fi/sipu/datasets/
2. Dummy clustering algorithm (random centroids and random partitions)
3. Distance function (Euclidean distance)
4. Sum-of-squared errors (SSE) calculation
"""

import numpy as np
import random
import urllib.request


class ClusteringTool:
    def __init__(self):
        self.data = None
        self.centroids = None
        self.partition = None
        self.k = 0
        
    def read_data_from_url(self, url):
        """
        Read data from URL (txt format from cs.uef.fi/sipu/datasets/)
        
        Args:
            url: URL to the dataset file
            
        Returns:
            numpy array of the data
        """
        try:
            print(f"Reading data from: {url}")
            response = urllib.request.urlopen(url)
            data_lines = response.read().decode('utf-8').strip().split('\n')
            
            # Parse data
            data_list = []
            for line in data_lines:
                # Skip empty lines
                if not line.strip():
                    continue
                # Parse numbers (space or tab separated)
                values = line.strip().split()
                data_list.append([float(v) for v in values])
            
            self.data = np.array(data_list)
            print(f"Data loaded: {self.data.shape[0]} points, {self.data.shape[1]} dimensions")
            return self.data
            
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
    
    def read_data_from_file(self, filepath):
        """
        Read data from local file (txt format)
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            numpy array of the data
        """
        try:
            print(f"Reading data from file: {filepath}")
            self.data = np.loadtxt(filepath)
            print(f"Data loaded: {self.data.shape[0]} points, {self.data.shape[1]} dimensions")
            return self.data
            
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
    
    def dummy_clustering(self, k):
        """
        Dummy clustering algorithm:
        (a) Select k random data points as centroids
        (b) Assign random partition labels (1 to k) to each data point
        
        Args:
            k: Number of clusters
        """
        if self.data is None:
            print("Error: No data loaded!")
            return
        
        self.k = k
        n_points = self.data.shape[0]
        
        # (a) Select k random data points as centroids
        random_indices = random.sample(range(n_points), k)
        self.centroids = self.data[random_indices].copy()
        
        # (b) Assign random partition labels (1 to k)
        self.partition = np.array([random.randint(1, k) for _ in range(n_points)])
        
        print(f"Dummy clustering completed: {k} clusters")
        print(f"Selected random centroids from indices: {random_indices}")
    
    def euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1: First data point (numpy array)
            point2: Second data point (numpy array)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def calculate_pairwise_distances(self):
        """
        Calculate pairwise distances between all data points
        and return the average distance
        
        Returns:
            Average pairwise distance
        """
        if self.data is None:
            print("Error: No data loaded!")
            return None
        
        n_points = self.data.shape[0]
        total_distance = 0
        count = 0
        
        print("Calculating pairwise distances...")
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distance = self.euclidean_distance(self.data[i], self.data[j])
                total_distance += distance
                count += 1
        
        avg_distance = total_distance / count
        print(f"Average pairwise distance: {avg_distance:.6f}")
        print(f"Total pairs calculated: {count}")
        
        return avg_distance
    
    def sum_of_squared_errors(self):
        """
        Calculate sum of squared errors (SSE)
        For each data point, calculate squared distance to ONE randomly chosen centroid
        
        Returns:
            SSE value
        """
        if self.data is None or self.centroids is None:
            print("Error: Data or centroids not available!")
            return None
        
        n_points = self.data.shape[0]
        sse = 0
        
        for i in range(n_points):
            # Randomly choose one centroid
            random_centroid_idx = random.randint(0, self.k - 1)
            random_centroid = self.centroids[random_centroid_idx]
            
            # Calculate squared distance
            squared_distance = np.sum((self.data[i] - random_centroid) ** 2)
            sse += squared_distance
        
        print(f"Sum of Squared Errors (SSE): {sse:.6f}")
        return sse
    
    def save_centroids(self, filename="centroid.txt"):
        """
        Save centroids to file
        Each centroid on its own line with attribute values separated by spaces
        
        Args:
            filename: Output filename for centroids
        """
        if self.centroids is None:
            print("Error: No centroids to save!")
            return
        
        with open(filename, 'w') as f:
            for centroid in self.centroids:
                line = ' '.join([str(val) for val in centroid])
                f.write(line + '\n')
        
        print(f"Centroids saved to: {filename}")
    
    def save_partition(self, filename="partition.txt"):
        """
        Save partition to file
        Each partition label on its own line
        
        Args:
            filename: Output filename for partition
        """
        if self.partition is None:
            print("Error: No partition to save!")
            return
        
        with open(filename, 'w') as f:
            for label in self.partition:
                f.write(str(label) + '\n')
        
        print(f"Partition saved to: {filename}")


def main():
    """
    Main function to demonstrate the clustering tool
    """
    print("=" * 60)
    print("Clustering Tool - Exercise 1")
    print("=" * 60)
    
    # Create clustering tool instance
    tool = ClusteringTool()
    
    # Example: Read S1 dataset from URL
    url = "http://cs.uef.fi/sipu/datasets/s1.txt"
    tool.read_data_from_url(url)
    
    # Set k (number of clusters)
    k = 15
    
    # Perform dummy clustering
    print("\n" + "=" * 60)
    tool.dummy_clustering(k)
    
    # Save results
    print("\n" + "=" * 60)
    tool.save_centroids("centroid.txt")
    tool.save_partition("partition.txt")
    
    # Calculate average pairwise distance
    print("\n" + "=" * 60)
    avg_dist = tool.calculate_pairwise_distances()
    
    # Calculate SSE with random centroid assignment
    print("\n" + "=" * 60)
    sse = tool.sum_of_squared_errors()
    
    print("\n" + "=" * 60)
    print("Clustering completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
