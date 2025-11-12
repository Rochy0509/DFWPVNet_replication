import torch
import numpy as np

class KeypointLocalizer:
    """
    Localize 2D keypoints using weighted voting based on potential field.
    Implements Algorithms 1 & 2 from DFW-PVNet paper.
    """
    
    def __init__(self, num_keypoints=9, num_samples=512, tau=0.999):
        self.K = num_keypoints
        self.N = num_samples
        self.tau = tau
    
    def localize(self, mask, vertex, potential):
        """
        Main entry point for keypoint localization.
        """
        # Convert to numpy if needed
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if torch.is_tensor(vertex):
            vertex = vertex.cpu().numpy()
        if torch.is_tensor(potential):
            potential = potential.cpu().numpy()
        
        # Generate hypotheses
        hypotheses = self.generate_hypotheses(mask, vertex, potential)
        
        # Weighted voting
        keypoints_init = self.weighted_voting(hypotheses, mask, vertex, potential)
        
        # Refinement (optional - simplified version)
        keypoints = self.refine_keypoints(keypoints_init, mask, vertex, potential)
        
        return keypoints
    
    def generate_hypotheses(self, mask, vertex, potential):
        """
        Algorithm 1: Keypoint-hypothesis generation.
        """
        H, W = mask.shape
        hypotheses = []
        
        # Get object pixel coordinates
        obj_pixels = np.argwhere(mask > 0)  # [num_pixels, 2] (y, x)
        
        for k in range(self.K):
            # Get potential values for this keypoint
            potential_k = potential[k] * mask  # [H, W]
            potential_flat = potential_k.flatten()
            
            # Select top N×2 pixels by potential
            num_samples = min(self.N * 2, len(obj_pixels))
            if num_samples < 2:
                # Not enough pixels, use center
                hypotheses.append(np.array([[W/2, H/2]]))
                continue
            
            top_indices = np.argpartition(potential_flat, -num_samples)[-num_samples:]
            
            # Convert to 2D coordinates
            coords_y = top_indices // W
            coords_x = top_indices % W
            coords = np.stack([coords_x, coords_y], axis=1)  # [num_samples, 2] (x, y)
            
            # Random split into two sets
            perm = np.random.permutation(num_samples)
            half = num_samples // 2
            indices_alpha = coords[perm[:half]]
            indices_beta = coords[perm[half:2*half]]
            
            # Get vectors for both sets
            vectors_alpha = np.stack([
                vertex[2*k, indices_alpha[:, 1], indices_alpha[:, 0]],    # vx
                vertex[2*k+1, indices_alpha[:, 1], indices_alpha[:, 0]]   # vy
            ], axis=1)  # [half, 2]
            
            vectors_beta = np.stack([
                vertex[2*k, indices_beta[:, 1], indices_beta[:, 0]],
                vertex[2*k+1, indices_beta[:, 1], indices_beta[:, 0]]
            ], axis=1)
            
            # Compute intersections
            hyps = self._compute_intersections(
                indices_alpha, vectors_alpha,
                indices_beta, vectors_beta
            )
            
            hypotheses.append(hyps)
        
        return hypotheses
    
    def _compute_intersections(self, coords_a, vectors_a, coords_b, vectors_b):
        """
        Compute intersection points of vector rays.
        """
        intersections = []
        
        for i in range(len(coords_a)):
            for j in range(len(coords_b)):
                # Point and direction for ray A
                p1 = coords_a[i]  # [2]
                d1 = vectors_a[i]  # [2]
                
                # Point and direction for ray B
                p2 = coords_b[j]
                d2 = vectors_b[j]
                
                # Solve for intersection: p1 + t1*d1 = p2 + t2*d2
                # Rearrange: t1*d1 - t2*d2 = p2 - p1
                A = np.column_stack([d1, -d2])
                b = p2 - p1
                
                # Check if lines are parallel
                det = np.linalg.det(A)
                if abs(det) < 1e-6:
                    continue
                
                # Solve for t1
                t = np.linalg.solve(A, b)
                t1 = t[0]
                
                # Compute intersection point
                intersection = p1 + t1 * d1
                intersections.append(intersection)
        
        if len(intersections) == 0:
            # Fallback to center
            return np.array([[coords_a[:, 0].mean(), coords_a[:, 1].mean()]])
        
        return np.array(intersections)
    
    def weighted_voting(self, hypotheses, mask, vertex, potential):
        """
        Algorithm 2: Weighted voting for keypoint location.
        """
        H, W = mask.shape
        keypoints = []
        
        # Get all object pixel coordinates
        obj_coords = np.argwhere(mask > 0)  # [num_pixels, 2] (y, x)
        if len(obj_coords) == 0:
            return np.zeros((self.K, 2))
        
        for k in range(self.K):
            best_score = -1
            best_hyp = None
            
            # Get vectors and potentials for this keypoint
            vertex_k = np.stack([
                vertex[2*k, obj_coords[:, 0], obj_coords[:, 1]],
                vertex[2*k+1, obj_coords[:, 0], obj_coords[:, 1]]
            ], axis=1)  # [num_pixels, 2]
            
            potential_k = potential[k, obj_coords[:, 0], obj_coords[:, 1]]  # [num_pixels]
            
            # Test each hypothesis
            for hyp in hypotheses[k]:
                score = self._vote_for_hypothesis(
                    hyp, obj_coords, vertex_k, potential_k
                )
                
                if score > best_score:
                    best_score = score
                    best_hyp = hyp
            
            if best_hyp is None:
                best_hyp = np.array([W/2, H/2])
            
            keypoints.append(best_hyp)
        
        return np.array(keypoints)
    
    def _vote_for_hypothesis(self, hypothesis, coords, vectors, potentials):
        """
        Calculate weighted voting score for a hypothesis.
        """
        # Calculate unit vectors from pixels to hypothesis
        # coords is (y, x), hypothesis is (x, y)
        coords_xy = np.stack([coords[:, 1], coords[:, 0]], axis=1)  # [N, 2] (x, y)
        diff = hypothesis - coords_xy  # [N, 2]
        
        # Normalize
        norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
        calculated_vectors = diff / norms  # [N, 2]
        
        # Cosine similarity
        similarity = np.sum(calculated_vectors * vectors, axis=1)  # [N]
        
        # Interior points (similarity > tau)
        interior_mask = similarity > self.tau
        
        # Weighted score
        weighted_scores = interior_mask.astype(float) * potentials
        score = np.sum(weighted_scores)
        
        return score
    
    def refine_keypoints(self, keypoints_init, mask, vertex, potential):
        """
        Refine keypoint positions using weighted least squares.
        Exact implementation following paper's Section 3.5.
        
        Paper's approach:
        1. For each keypoint, get interior points (inners)
        2. Compute normals (perpendicular to predicted unit vectors)
        3. Build weighted least squares problem
        4. Solve: keypoints = (normal x M^T M x normal^T)^-1 
                            x normal x M^T M x coords^T x normal
        """
        H, W = mask.shape
        keypoints_refined = []
        
        # Get all object pixel coordinates
        obj_coords = np.argwhere(mask > 0)  # [N, 2] (y, x)
        if len(obj_coords) == 0:
            return keypoints_init
        
        # Convert to (x, y) format
        coords_xy = np.stack([obj_coords[:, 1], obj_coords[:, 0]], axis=1)  # [N, 2]
        
        for k in range(self.K):
            keypoint_init = keypoints_init[k]  # [2] (x, y)
            
            # Step 1: Get predicted unit vectors (directs)
            directs = np.stack([
                vertex[2*k, obj_coords[:, 0], obj_coords[:, 1]],
                vertex[2*k+1, obj_coords[:, 0], obj_coords[:, 1]]
            ], axis=1)  # [N, 2] (vx, vy)
            
            # Step 2: Calculate directs' (from pixels to keypoint)
            diff = keypoint_init - coords_xy
            norms = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
            directs_prime = diff / norms  # [N, 2]
            
            # Step 3: Find interior points (cosine similarity > tau)
            similarity = np.sum(directs * directs_prime, axis=1)
            interior_mask = similarity > self.tau
            
            if np.sum(interior_mask) < 3:
                keypoints_refined.append(keypoint_init)
                continue
            
            # Interior points data
            coords = coords_xy[interior_mask]  # [M, 2]
            directs_inners = directs[interior_mask]  # [M, 2]
            
            # Get potential weights
            potential_k = potential[k, obj_coords[:, 0], obj_coords[:, 1]]
            potentials = potential_k[interior_mask]  # [M]
            
            # Step 4: Compute normals (perpendicular to directs)
            # For 2D: normal of [vx, vy] is [-vy, vx]
            normals = np.stack([
                -directs_inners[:, 1],
                directs_inners[:, 0]
            ], axis=1)  # [M, 2]
            
            # Step 5: Build weighted matrix M
            M = np.diag(np.sqrt(potentials))  # [M, M]
            
            # Step 6: Solve WLS
            # The goal is to find keypoint position where:
            # normals · (keypoint - coords) = 0 (weighted) almost
            
            try:
                # Build system: A × keypoint = b
                MTM = M.T @ M  # [M, M]
                
                # A = normals^T × M^T M × normals
                A = normals.T @ MTM @ normals  # [2, 2]
                
                # b = normals^T × M^T M × (normals ⊙ coords).sum()
                # where ⊙ is element-wise multiplication
                
                # Each normal should satisfy: normal_i · keypoint = normal_i · coords_i
                rhs = (normals * coords).sum(axis=1)  # [M]
                b = normals.T @ MTM @ rhs  # [2]
                
                # Solve A × keypoint = b
                if np.abs(np.linalg.det(A)) < 1e-6:
                    keypoints_refined.append(keypoint_init)
                    continue
                
                refined_keypoint = np.linalg.solve(A, b)  # [2]
                
                # Sanity check
                distance = np.linalg.norm(refined_keypoint - keypoint_init)
                if distance > 100:  # More than 100 pixels away
                    keypoints_refined.append(keypoint_init)
                else:
                    keypoints_refined.append(refined_keypoint)
                    
            except np.linalg.LinAlgError:
                keypoints_refined.append(keypoint_init)
        
        return np.array(keypoints_refined)