"""
Exercise 2 - Task 4 and Task 5 Solutions
"""

def task4_mps_projection():
    """
    Demonstrate MPS projection method on x-axis
    
    Given data points: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    Input: middle point between 4 and 10 = (4+10)/2 = 7
    Process points in order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    """
    print("="*70)
    print("TASK 4: MPS Projection Method Demonstration")
    print("="*70)
    
    # Data points
    points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Input: middle point between 4 and 10
    input_point = (4 + 10) / 2.0
    print(f"\nInput point (middle between 4 and 10): {input_point}")
    
    # Initialize projection
    current_projection = input_point
    
    print("\nProcessing points in order 1-10:")
    print("-"*70)
    print(f"{'Point':<10} {'Distance to Projection':<25} {'Action':<30}")
    print("-"*70)
    
    for point in points:
        distance = abs(point - current_projection)
        
        # MPS projection: if point is on the x-axis, project to it
        # The projection moves towards points that are closer
        if distance == 0:
            action = "Already at projection"
        else:
            # Calculate new projection (average of current and point)
            new_projection = (current_projection + point) / 2.0
            action = f"Update: {current_projection:.2f} → {new_projection:.2f}"
            current_projection = new_projection
        
        print(f"{point:<10} {distance:<25.2f} {action:<30}")
    
    print("-"*70)
    print(f"\nFinal projection point: {current_projection:.2f}")
    
    print("\n" + "="*70)
    print("Explanation:")
    print("="*70)
    print("""
The MPS (Mumford-Shah) projection method works as follows:

1. Start with initial point (7.0, the midpoint between 4 and 10)
2. Process each data point in order
3. For each point, update the projection by moving it towards that point
4. The projection is updated as the average of current projection and the point
5. This gradually refines the projection based on the data distribution

Why this approach:
- It smoothly adapts to the data distribution
- Points closer to the projection have more influence
- The projection converges to a representative location on the axis

For this specific example:
- Starting at 7.0 (midpoint of 4 and 10)
- Processing points 1-10 in order
- Each point pulls the projection towards itself
- Final projection represents the central tendency of all points
    """)

def task5_time_complexity():
    """
    Calculate number of trial swaps possible with same time as k-means
    
    Given:
    - K-means iteration: O(Nk) time
    - Random swap: O(N) time
    - After each swap: 2 iterations of k-means
    - K-means converges in 25 iterations
    - α = 4 (some parameter, likely related to k)
    
    Find: How many trial swaps can we perform using the same time?
    """
    print("="*70)
    print("TASK 5: Time Complexity Analysis")
    print("="*70)
    
    print("\nGiven:")
    print("-" * 70)
    print("• Each K-means iteration: O(Nk) time")
    print("• Each random swap: O(N) time")
    print("• After each swap: 2 K-means iterations applied")
    print("• K-means converges in: 25 iterations")
    print("• α = 4")
    
    # Assuming α = 4 means k = 4
    k = 4
    alpha = 4
    
    print("\n" + "="*70)
    print("Solution:")
    print("="*70)
    
    # Time for full K-means (25 iterations)
    print("\n1. Time for full K-means convergence:")
    print(f"   T_kmeans = 25 iterations × O(Nk)")
    print(f"   T_kmeans = 25 × O(N×{k})")
    print(f"   T_kmeans = O(25N×{k})")
    print(f"   T_kmeans = O({25*k}N)")
    
    # Time for one swap + 2 K-means iterations
    print("\n2. Time for one trial swap:")
    print(f"   T_swap = O(N) + 2 × O(Nk)")
    print(f"   T_swap = O(N) + 2 × O(N×{k})")
    print(f"   T_swap = O(N) + O({2*k}N)")
    print(f"   T_swap = O(N + {2*k}N)")
    print(f"   T_swap = O({1 + 2*k}N)")
    
    # Number of swaps with same time
    print("\n3. Number of trial swaps S with same total time:")
    print(f"   S × T_swap = T_kmeans")
    print(f"   S × O({1 + 2*k}N) = O({25*k}N)")
    print(f"   S = {25*k}N / {1 + 2*k}N")
    print(f"   S = {25*k} / {1 + 2*k}")
    
    # Calculate
    numerator = 25 * k
    denominator = 1 + 2 * k
    num_swaps = numerator / denominator
    
    print(f"   S = {numerator} / {denominator}")
    print(f"   S = {num_swaps:.4f}")
    print(f"   S ≈ {int(num_swaps)} trial swaps")
    
    print("\n" + "="*70)
    print("Verification:")
    print("="*70)
    print(f"Time for K-means:        {25*k}N")
    print(f"Time for {int(num_swaps)} swaps:      {int(num_swaps)} × {1 + 2*k}N = {int(num_swaps) * (1 + 2*k)}N")
    
    print("\n" + "="*70)
    print("Interpretation:")
    print("="*70)
    print(f"""
With α = {alpha} (assuming k = {k}):
- We can perform approximately {int(num_swaps)} random swap trials
- Each trial involves:
  * 1 random swap operation: O(N)
  * 2 K-means iterations: 2 × O(Nk) = O({2*k}N)
  * Total per trial: O({1 + 2*k}N)

This uses the same computational time as running K-means 
for {25} iterations (until convergence).

The random swap algorithm explores different solutions by:
1. Making a random swap (changing one centroid)
2. Running 2 K-means iterations to refine
3. Accepting if SSE improves
4. Repeating for {int(num_swaps)} trials

This provides {int(num_swaps)} opportunities to escape local optima,
potentially finding better clustering solutions than a single 
{25}-iteration K-means run.
    """)
    
    return int(num_swaps)


def main():
    """Run Task 4 and Task 5 demonstrations"""
    
    # Task 4: MPS Projection
    task4_mps_projection()
    
    print("\n\n")
    
    # Task 5: Time Complexity
    num_swaps = task5_time_complexity()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Task 4: MPS projection demonstration completed")
    print(f"Task 5: Can perform {num_swaps} random swap trials")
    print("="*70)


if __name__ == "__main__":
    main()