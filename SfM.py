import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation 
import scipy.optimize as optimize
from scipy.sparse import lil_matrix
import time
import matplotlib.pyplot as plt
import open3d as o3d
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D

def normalize(uv):
    uv_ = np.mean(uv, axis=0)
    u_,v_ = uv_[0], uv_[1]
    u_cap, v_cap = uv[:,0] - u_, uv[:,1] - v_

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_],[0,1,-v_],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))  #[x,y,1]
    x_norm = (T.dot(x_.T)).T     #x_ = T.x

    return  x_norm, T

def EstimateFundamentalMatrix(pts1, pts2):
    normalised = True

    x1,x2 = pts1, pts2

    if x1.shape[0] > 7:
        if normalised == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))     #Fundamental matrix 3x3 so 9columns and min 8 points So rows>=8 columns = 9
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0                     #Due to Noise F can be full rank i.e 3, but we need to make it rank 2 by assigning zero to last diagonal element and thus we get the epipoles
        F = np.dot(u, np.dot(s, vt))
        
        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))   #This is given in algorithm for normalization
            F = F / F[2,2]
        return F

    else:
        return None


def error_F(pts1,pts2,F):
    
    "Checking the epipolar constraint"
    x1 = np.array([pts1[0], pts1[1],1])
    x2 = np.array([pts2[0], pts2[1],1]).T

    error = np.dot(x2,np.dot(F,x1))

    return np.abs(error)

def getInliers(pts1,pts2,idx):

    "Point Correspondence are computed using SIFT feature descriptors, data becomes noisy, RANSAC is used with fundamental matrix with maximum no of Inliers"
    
    no_iterations = 2000
    error_threshold = 0.002
    inliers_threshold = 0
    inliers_indices = []
    f_inliers = None

    for i in range(0, no_iterations):
        # We need 8 points randomly for 8 points algorithm
        n_rows = pts1.shape[0]
        rand_indxs = np.random.choice(n_rows,8)
        x1 = pts1[rand_indxs,:]
        x2 = pts2[rand_indxs,:]
        F = EstimateFundamentalMatrix(x1,x2)
        indices = []

        if F is not None:
            for j in range(n_rows):
                error = error_F(pts1[j,:],pts2[j,:],F)  #x2.TFx1 = 0
                if error < error_threshold:
                    indices.append(idx[j])

        if len(indices) > inliers_threshold:
            inliers_threshold = len(indices)
            inliers_indices = indices
            f_inliers = F                       #We choose F with Maximum no of Inliers.

    return F, inliers_indices

def getEssentialMatrix(K,F):
    E = K.T.dot(F).dot(K)
    U,S,V = np.linalg.svd(E)
    S = [1,1,0]

    return np.dot(U,np.dot(np.diag(S),V))


def ExtractCameraPose(E):
    """
    E (array) - Essential Matrix
    K (array) - Intrinsic Matrix

    Returns - 4 Sets of Rotation and Camera centers
    """

    U,S,V_T = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    R = []
    C = []
    R.append(np.dot(U,np.dot(W,V_T)))
    R.append(np.dot(U,np.dot(W,V_T)))
    R.append(np.dot(U,np.dot(W.T,V_T)))
    R.append(np.dot(U,np.dot(W.T,V_T)))
    C.append(U[:,2])
    C.append(-U[:,2])
    C.append(U[:,2])
    C.append(-U[:,2])

    for i in range(4):
        if (np.linalg.det(R[i])<0):
            R[i] = -R[i]
            C[i] = -C[i]


    return R, C

def linearTriangulation(K, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    C1 = np.reshape(C1, (3,1))
    C2 = np.reshape(C2, (3,1))

    P1 = np.dot(K, np.dot(R1, np.hstack((I,-C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I,-C2))))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)

    X = []
    for i in range(x1.shape[0]):
       x = x1[i,0] 
       y = x1[i,1]
       x_ = x2[i,0]
       y_ = x2[i,1]

       A = []
       A.append((y * p3T) - p2T)
       A.append(p1T - (x * p3T))
       A.append((y_ * p_3T) - p_2T)
       A.append(p_1T - (x_ * p_3T))

       A = np.array(A).reshape(4,4)

       _,_,vt = np.linalg.svd(A)
       v = vt.T
       x = v[:,-1]
       X.append(x)

    return np.array(X)

def plot_triangulation(X, C_set, title, output_dir='Outputs', file_name='triangulation.png'):
    ensure_dir(output_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='o', label='3D Points')
    
    # Plot camera positions
    for i, C in enumerate(C_set):
        ax.scatter(C[0], C[1], C[2], c='r', marker='^', label=f'Camera {i+1}' if i == 0 else "")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.savefig(os.path.join(output_dir, file_name))
    plt.show()


def DisambiguatePose(r_set, c_set, x3D_set):

    best_i = 0
    max_positive_depths = 0

    for i in range(len(r_set)):
        R, C = r_set[i], c_set[i]
        r3 = R[2, :].reshape(1,-1) #3rd column of R
        x3D = x3D_set[i]
        x3D = x3D / x3D[:,3].reshape(-1,1)
        x3D = x3D[:, 0:3]

        #Here we count MAximum Positive Depths
        n_positive_depths = DepthPositivityConstraint(x3D, r3,C)
        # print(n_positive_depths)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths

    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]

    return R, C, x3D 

def DepthPositivityConstraint(x3D, r3, C):
    # r3(X-C) alone doesnt solve the check positivity. z = X[2] must also be +ve 
    n_positive_depths=  0
    for X in x3D:
        X = X.reshape(-1,1) 
        C = C.reshape(-1,1)
        if r3.dot(X-C).T>0 and X[2]>0: 
            n_positive_depths+=1
    return n_positive_depths

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P



def NonLinearTriangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    """
    K : Camera Matrix
    pts1, pts2 : Point Correspondences
    x3D :  initial 3D point
    R2, C2 : relative camera pose
    Returns:
        x3D : optimized 3D points
    """

    P1 = ProjectionMatrix(R1,C1,K) 
    P2 = ProjectionMatrix(R2,C2,K)
    
    if pts1.shape[0] != pts2.shape[0] != x3D.shape[0]:
        raise 'Check point dimensions - level nlt'

    x3D_ = []
    for i in range(len(x3D)):
        optimized_params = optimize.least_squares(fun=ReprojectionLoss, x0=x3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2])
        X1 = optimized_params.x
        x3D_.append(X1)
    return np.array(x3D_)


def ReprojectionLoss(X, pts1, pts2, P1, P2):
    
    # X = homo(X.reshape(1,-1)).reshape(-1,1) # make X a column of homogenous vector
    
    p1_1T, p1_2T, p1_3T = P1 # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = P2 # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points - j = 1
    u1,v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    
    ## reprojection error for second camera points - j = 2    
    u2,v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    error = E1 + E2
    return error.squeeze()

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))
    
def reprojectionErrorPnP(x3D, pts, K, R, C):
    P = ProjectionMatrix(R,C,K)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    mean_error = np.mean(np.array(Error).squeeze())
    return mean_error


def PnP(X_set, x_set, K):
    N = X_set.shape[0]
    
    X_4 = homo(X_set)
    x_3 = homo(x_set)
    
    # normalize x
    K_inv = np.linalg.inv(K)
    x_n = K_inv.dot(x_3.T).T
    
    for i in range(N):
        X = X_4[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        
        u, v, _ = x_n[i]
        
        u_cross = np.array([[0, -1, v],
                            [1,  0 , -u],
                            [-v, u, 0]])
        X_tilde = np.vstack((np.hstack((   X, zeros, zeros)), 
                            np.hstack((zeros,     X, zeros)), 
                            np.hstack((zeros, zeros,     X))))
        a = u_cross.dot(X_tilde)
        
        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a
            
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, D, V_rT = np.linalg.svd(R) # to enforce Orthonormality
    R = U_r.dot(V_rT)
    
    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
        
    return R, C

def plot_acamera_poses(R_set, C_set, output_dir='Outputs', file_name='camera_poses.png'):

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin
    ax.scatter(0, 0, 0, c='k', marker='o', label='Origin')


    # Plot camera poses
    for i, (R, C) in enumerate(zip(R_set, C_set)):
        # Convert rotation matrix to Euler angles for better visualization
        euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        ax.quiver(0, 0, 0, 0.2*C[0], 0.2*C[1], 0.2*C[2], color='r', label=f'Camera {i+1}: Roll={euler_angles[0]:.1f}, Pitch={euler_angles[1]:.1f}, Yaw={euler_angles[2]:.1f}')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(output_dir, file_name))
    plt.show()


def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def PnPError(feature, X, R, C, K):
    u,v = feature
    pts = X.reshape(1,-1)
    X = np.hstack((pts, np.ones((pts.shape[0],1))))
    X = X.reshape(4,1)
    C = C.reshape(-1,1)
    P = ProjectionMatrix(R,C,K)
    p1, p2, p3 = P
    p1, p2, p3 = p1.reshape(1,4), p2.reshape(1,4), p3.reshape(1,4)
    u_proj = np.divide(p1.dot(X), p3.dot(X))
    v_proj = np.divide(p2.dot(X), p3.dot(X))

    x_proj = np.hstack((u_proj, v_proj))
    x = np.hstack((u,v))
    err = np.linalg.norm(x-x_proj)

    return err

def PnPRANSAC(K,features,x3D, iter=1000, thresh=5):
    inliers_thresh = 0
    R_best, t_best = None, None
    n_rows = x3D.shape[0]

    for i in range(0, iter):

        #We rand select 6 pts
        rand_indices = np.random.choice(n_rows, size=6)
        X_set, x_set = x3D[rand_indices], features[rand_indices]

        #WE get R and C from PnP function
        R, C = PnP(X_set, x_set, K)

        indices = []
        if R is not None:
            for j in range(n_rows):
                feature = features[j]
                X = x3D[j]
                error = PnPError(feature, X, R, C, K)

                if error < thresh:
                    indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            R_best = R
            t_best = C

    return R_best, t_best

def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def getQuaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def NonLinearPnP(K, pts, x3D, R0, C0):
    """
    K : Camera Matrix
    pts1, pts2 : Point Correspondences
    x3D :  initial 3D point
    R2, C2 : relative camera pose - estimated from PnP
    Returns:
        x3D : optimized 3D points
    """

    Q = getQuaternion(R0)
    X0 = [Q[0] ,Q[1],Q[2],Q[3], C0[0], C0[1], C0[2]] 

    optimized_params = optimize.least_squares(
        fun = PnPLoss,
        x0=X0,
        method="trf",
        args=[x3D, pts, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = getRotation(Q)
    return R, C

def PnPLoss(X0, x3D, pts, K):
    
    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = getRotation(Q)
    P = ProjectionMatrix(R,C,K)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)


        X = homo(X.reshape(1,-1)).reshape(-1,1) 
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    sumError = np.mean(np.array(Error).squeeze())
    return sumError
def getObservationsIndexAndVizMat(X_found, filtered_feature_flag, nCam):
    # find the 3d points such that they are visible in either of the cameras < nCam
    bin_temp = np.zeros((filtered_feature_flag.shape[0]), dtype = int)
    for n in range(nCam + 1):
        bin_temp = bin_temp | filtered_feature_flag[:,n]

    X_index = np.where((X_found.reshape(-1)) & (bin_temp))
    
    visiblity_matrix = X_found[X_index].reshape(-1,1)
    for n in range(nCam + 1):
        visiblity_matrix = np.hstack((visiblity_matrix, filtered_feature_flag[X_index, n].reshape(-1,1)))

    o, c = visiblity_matrix.shape
    return X_index, visiblity_matrix[:, 1:c]

def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def getEuler(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()


def get2DPoints(X_index, visiblity_matrix, feature_x, feature_y):
    "Get 2D Points from the feature x and feature y having same index from the visibility matrix"
    pts2D = []
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2) 

def getCameraPointIndices(visiblity_matrix):

    "From Visibility Matrix take away indices of point visible from Camera pose by taking indices of cam as well"

    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)

def bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam):
    
    "Create Sparsity Matrix"

    number_of_cam = nCam + 1
    X_index, visiblity_matrix = getObservationsIndexAndVizMat(X_found.reshape(-1), filtered_feature_flag, nCam)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3   #Here we don't take focal length and 2 radial distortion parameters, i.e no - 6. We just refine orientation and translation of 3d point and not cam parameters.
    A = lil_matrix((m, n), dtype=int)
    # print(m, n)


    i = np.arange(n_observations)
    camera_indices, point_indices = getCameraPointIndices(visiblity_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A    

def project(points_3d, camera_params, K):
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = getRotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def fun(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec  


def BundleAdjustment(X_index,visibility_matrix,X_all,X_found,feature_x,feature_y, filtered_feature_flag, R_set, C_set, K, nCam):
    points_3d = X_all[X_index]
    points_2d = get2DPoints(X_index,visibility_matrix,feature_x,feature_y)

    RC = []
    for i in range(nCam+1):
        C, R = C_set[i], R_set[i]
        Q = getEuler(R)
        RC_ = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC.append(RC_)
    RC = np.array(RC, dtype=object).reshape(-1,6)

    x0 = np.hstack((RC.ravel(), points_3d.ravel()))
    n_pts = points_3d.shape[0]

    camera_indices, points_indices = getCameraPointIndices(visibility_matrix)

    A = bundle_adjustment_sparsity(X_found,filtered_feature_flag,nCam)
    t0 = time.time()
    res = least_squares(fun,x0,jac_sparsity=A, verbose=2,x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, n_pts, camera_indices, points_indices, points_2d,K))

    t1 = time.time()
    print("Time required to run Bundle Adj: ", t1-t0, "s \nA matrix shape: ",A.shape,"\n######")

    x1 = res.x
    no_of_cams = nCam + 1
    optim_cam_param = x1[:no_of_cams*6].reshape((no_of_cams,6))
    optim_pts_3d = x1[no_of_cams*6:].reshape((n_pts,3))

    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_pts_3d

    optim_C_set , optim_R_set = [], []
    for i in range(len(optim_cam_param)):
        R = getRotation(optim_cam_param[i,:3], 'e')
        C = optim_cam_param[i,3:].reshape(3,1)
        optim_C_set.append(C)
        optim_R_set.append(R)

    return optim_R_set, optim_C_set, optim_X_all
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_2d_trajectory(X, C_set, R_set, output_dir='Outputs'):
    ensure_dir(output_dir)
    x = X[:, 0]
    z = X[:, 2]

    fig = plt.figure(figsize=(10, 10))
    plt.xlim(-4, 6)
    plt.ylim(-2, 12)
    plt.scatter(x, z, marker='.', linewidths=0.5, color='blue')
    for i in range(len(C_set)):
        R1 = getEuler(R_set[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set[i][0], C_set[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig(os.path.join(output_dir, '2D.png'))
    plt.show()

def plot_3d_trajectory(X, C_set, output_dir='Outputs'):
    ensure_dir(output_dir)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    fig1 = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z, color="green")
    plt.savefig(os.path.join(output_dir, '3D.png'))
    plt.show()

def plot_camera_poses(R_set, T_set, X, output_dir='Outputs'):
    ensure_dir(output_dir)
    colormap = ['r', 'b', 'g', 'y']
    for i in range(len(R_set)):
        C = T_set[i]
        R = R_set[i]
        X = point3D_set[i]
        R = Rotation.from_matrix(R).as_rotvec()
        R1 = np.rad2deg(R)
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(R1[1]))
        ax = plt.gca()
        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
        ax.scatter(X[:, 0], X[:, 2], s=4, color=colormap[i], label='cam'+str(i))
    plt.xlim(-20, 20)
    plt.ylim(-20, 30)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'all_poses.png'))
    plt.show()

def plot_poses_3d(R_set, T_set, point3D_set, output_dir='Outputs'):
    ensure_dir(output_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = ['r', 'b', 'g', 'y']
    
    for i in range(len(R_set)):
        C = T_set[i]
        R = R_set[i]
        X = point3D_set[i]
        
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=colormap[i], label=f'cam{i}')
        ax.scatter(C[0], C[1], C[2], marker='^', s=100, color=colormap[i])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'all_poses_3d.png'))
    plt.show()

def plot_selected_pose_3d(T_best, R_best, X_, index, output_dir='Outputs'):
    ensure_dir(output_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = ['r', 'b', 'g', 'y']
    
    C = T_best
    R = R_best
    X = X_
    
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=colormap[index], label=f'cam{index}')
    ax.scatter(C[0], C[1], C[2], marker='^', s=100, color=colormap[index])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'selected_pose_3d.png'))
    plt.show()

# Define other plotting functions here

# Function to convert rotation matrix to Euler angles
def getEuler(R):
    return Rotation.from_matrix(R).as_euler('xyz', degrees=True)


def features_extraction(data):
    no_of_images = 5
    feature_rgb_values = []
    feature_x = []
    feature_y = []
    feature_flag = []

    for n in range(1, no_of_images):
        file = data + "/matching" + str(n) + ".txt"
        matching_file = open(file, "r")


        for i, row in enumerate(matching_file):
            if i == 0:
                row_elements = row.split(':')

            else:
                x_row = np.zeros((1, no_of_images))
                y_row = np.zeros((1, no_of_images))
                flag_row = np.zeros((1, no_of_images), dtype=int)
                row_elements = row.split()
                columns = [float(x) for x in row_elements]
                columns = np.asarray(columns)

                nMatches = columns[0]
                r_value = columns[1]
                b_value = columns[2]
                g_value = columns[3]

                feature_rgb_values.append([r_value, g_value, b_value])
                current_x = columns[4]
                current_y = columns[5]

                x_row[0, n - 1] = current_x
                y_row[0, n - 1] = current_y
                flag_row[0, n - 1] = 1

                m = 1
                while nMatches > 1:
                    image_id = int(columns[5 + m])
                    image_id_x = int(columns[6 + m])
                    image_id_y = int(columns[7 + m])
                    m = m + 3
                    nMatches = nMatches - 1

                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = 1

                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)

    feature_x = np.asarray(feature_x).reshape(-1, no_of_images)
    feature_y = np.asarray(feature_y).reshape(-1, no_of_images)
    feature_flag = np.asarray(feature_flag).reshape(-1, no_of_images)
    feature_rgb_values = np.asarray(feature_rgb_values).reshape(-1, 3)

    return feature_x, feature_y, feature_flag, feature_rgb_values

def projectionMatrix(R, C, K):
    C = np.reshape(C, (3, 1))
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def ReProjectionError(X, pt1, pt2, R1, C1, R2, C2, K):
    p1 = projectionMatrix(R1, C1, K)
    p2 = projectionMatrix(R2, C2, K)

    p1_1T, p1_2T, p1_3T = p1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1, 4), p1_2T.reshape(1, 4), p1_3T.reshape(1, 4)

    p2_1T, p2_2T, p2_3T = p2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1, 4), p2_2T.reshape(1, 4), p2_3T.reshape(1, 4)

    X = X.reshape(4, 1)

    u1, v1 = pt1[0], pt1[1]
    u1_projection = np.divide(p1_1T.dot(X), p1_3T.dot(X))
    v1_projection = np.divide(p1_2T.dot(X), p1_3T.dot(X))
    err1 = np.square(v1 - v1_projection) + np.square(u1 - u1_projection)

    u2, v2 = pt2[0], pt2[1]
    u2_projection = np.divide(p2_1T.dot(X), p2_3T.dot(X))
    v2_projection = np.divide(p2_2T.dot(X), p2_3T.dot(X))
    err2 = np.square(v2 - v2_projection) + np.square(u2 - u2_projection)

    return err1, err2

def open3D_vis():
    pass

def main():
    Parser = argparse.ArgumentParser()
    # Specify the full path to the Outputs directory
    Parser.add_argument('--Outputs', default='D:/ENPM673_project5_test/Output', help='Outputs are saved here')
    # Specify the full path to the Data directory
    Parser.add_argument('--Data', default='D:/ENPM673_project5_test/Dataset/', help='Data')

    Args = Parser.parse_args()
    Data = Args.Data
    Output = Args.Outputs

    # Create the output directory if it does not exist
    if not os.path.exists(Output):
        os.makedirs(Output)

    images = []
    for i in range(1, 6):
        path = Data + "/" + str(i) + ".png"
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print("No image is found")

    feature_x, feature_y, feature_flag, feature_rgb_values = features_extraction(Data)
    filtered_feature_flag = np.zeros_like(feature_flag)
    f_matrix = np.empty(shape=(5, 5), dtype=object)

    for i in range(0, 4):
        for j in range(i + 1, 5):
            idx = np.where(feature_flag[:, i] & feature_flag[:, j])
            pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
            pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
            idx = np.array(idx).reshape(-1)

            if len(idx) > 8:
                F_inliers, inliers_idx = getInliers(pts1, pts2, idx)
                print("Between Images: ", i, "and", j, "NO of Inliers: ", len(inliers_idx), "/", len(idx))
                f_matrix[i, j] = F_inliers
                filtered_feature_flag[inliers_idx, j] = 1
                filtered_feature_flag[inliers_idx, i] = 1

    print("######Obtained Feature Points after RANSAC#######")
    print("Starting with 1st 2 images")

    F12 = f_matrix[0, 1]
    K = np.array([[531.122155322710, 0, 407.192550839899], [0, 531.541737503901, 313.308715048366], [0, 0, 1]])
    E12 = getEssentialMatrix(K, F12)
    print("essential_matrix_found")
    R_set, C_set = ExtractCameraPose(E12)
    plot_acamera_poses(R_set, C_set, output_dir=Output, file_name='ambiguous_camera_poses.png')
    idx = np.where(filtered_feature_flag[:, 0] & filtered_feature_flag[:, 1])
    pts1 = np.hstack((feature_x[idx, 0].reshape((-1, 1)), feature_y[idx, 0].reshape((-1, 1))))
    pts2 = np.hstack((feature_x[idx, 1].reshape((-1, 1)), feature_y[idx, 1].reshape((-1, 1))))

    R1_ = np.identity(3)
    C1_ = np.zeros((3, 1))
    print(len(C_set))

    pts3D_4 = []
    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        X = linearTriangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)
        print("Triangulation done!")

        X = X / X[:, 3].reshape(-1, 1)
        pts3D_4.append(X)

    R_best, C_best, X = DisambiguatePose(R_set, C_set, pts3D_4)
    X = X / X[:, 3].reshape(-1, 1)
    X_linear = linearTriangulation(K, C1_, R1_, C_best, R_best, pts1, pts2)
    X_linear_homogeneous = X_linear / X_linear[:, 3].reshape(-1, 1)  # Convert to homogeneous coordinates
    plot_triangulation(X_linear_homogeneous[:, :3], [C1_, C_best], 'Linear Triangulation', output_dir=Output, file_name='linear_triangulation.png')

    print("######Performing Non-Linear Triangulation######")
    X_refined = NonLinearTriangulation(K, pts1, pts2, X, R1_, C1_, R_best, C_best)
    X_refined = X_refined / X_refined[:, 3].reshape(-1, 1)

    # Call the plotting function here
    plot_triangulation(X_refined[:, :3], [C1_, C_best], 'Nonlinear Triangulation', output_dir=Output, file_name='nonlinear_triangulation.png')
    total_err1 = []
    for pt1, pt2, X_3d in zip(pts1, pts2, X):
        err1, err2 = ReProjectionError(X_3d, pt1, pt2, R1_, C1_, R_best, C_best, K)
        total_err1.append(err1 + err2)

    mean_err1 = np.mean(total_err1)

    total_err2 = []
    for pt1, pt2, X_3d in zip(pts1, pts2, X_refined):
        err1, err2 = ReProjectionError(X_3d, pt1, pt2, R1_, C1_, R_best, C_best, K)
        total_err2.append(err1 + err2)

    mean_err2 = np.mean(total_err2)

    print("Between images", 0 + 1, 1 + 1, "Before optimization Linear Triang: ", mean_err1, "After optimization Non-Linear Triang: ", mean_err2)

    X_all = np.zeros((feature_x.shape[0], 3))
    cam_indices = np.zeros((feature_x.shape[0], 1), dtype=int)
    X_found = np.zeros((feature_x.shape[0], 1), dtype=int)
    X_all[idx] = X[:, :3]
    X_found[idx] = 1
    cam_indices[idx] = 1
    X_found[np.where(X_all[:2] < 0)] = 0

    C_set = []
    R_set = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set.append(C0)
    R_set.append(R0)
    C_set.append(C_best)
    R_set.append(R_best)

    print("#########Registered Cam 1 and Cam 2 ############")

    for i in range(2, 5):
        print("Registering Image: ", str(i + 1))
        feature_idx_i = np.where(X_found[:, 0] & filtered_feature_flag[:, i])
        if len(feature_idx_i[0]) < 8:
            print("Got ", len(feature_idx_i), "common points between X and ", i, "image")
            continue

        pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1, 1), feature_y[feature_idx_i, i].reshape(-1, 1)))
        X = X_all[feature_idx_i, :].reshape(-1, 3)

        R_init, C_init = PnPRANSAC(K, pts_i, X, iter=1000, thresh=5)
        linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, R_init, C_init)

        Ri, Ci = NonLinearPnP(K, pts_i, X, R_init, C_init)
        non_linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, Ri, Ci)
        print("Initial linear PnP error: ", linear_error_pnp, " Final Non-linear PnP error: ", non_linear_error_pnp)

        C_set.append(Ci)
        R_set.append(Ri)

        for k in range(0, i):
            idx_X_pts = np.where(filtered_feature_flag[:, k] & filtered_feature_flag[:, i])
            idx_X_pts = np.asarray(idx_X_pts)
            idx_X_pts = np.squeeze(idx_X_pts)

            if len(idx_X_pts) < 8:
                continue

            x1 = np.hstack((feature_x[idx_X_pts, k].reshape(-1, 1), feature_y[idx_X_pts, k].reshape(-1, 1)))
            x2 = np.hstack((feature_x[idx_X_pts, i].reshape(-1, 1), feature_y[idx_X_pts, i].reshape(-1, 1)))

            X_d = linearTriangulation(K, C_set[k], R_set[k], Ci, Ri, x1, x2)
            X_d = X_d / X_d[:, 3].reshape(-1, 1)

            linear_err = []
            pts1, pts2 = x1, x2
            for pt1, pt2, X_3d in zip(pts1, pts2, X_d):
                err1, err2 = ReProjectionError(X_3d, pt1, pt2, R_set[k], C_set[k], Ri, Ci, K)
                linear_err.append(err1 + err2)

            mean_linear_err = np.mean(linear_err)

            X = NonLinearTriangulation(K, x1, x2, X_d, R_set[k], C_set[k], Ri, Ci)
            X = X / X[:, 3].reshape(-1, 1)

            non_linear_err = []
            for pt1, pt2, X_3d in zip(pts1, pts2, X):
                err1, err2 = ReProjectionError(X_3d, pt1, pt2, R_set[k], C_set[k], Ri, Ci, K)
                non_linear_err.append(err1 + err2)

            mean_nonlinear_err = np.mean(non_linear_err)
            print("Linear Triang error: ", mean_linear_err, "Non-linear Triang error: ", mean_nonlinear_err)

            X_all[idx_X_pts] = X[:, :3]
            X_found[idx_X_pts] = 1

            print("Appended", idx_X_pts[0], "Points Between ", k, "and ", i)

            X_index, visibility_matrix = getObservationsIndexAndVizMat(X_found, filtered_feature_flag, nCam=i)

            print("########Bundle Adjustment Started")
            R_set_, C_set_, X_all = BundleAdjustment(X_index, visibility_matrix, X_all, X_found, feature_x, feature_y, filtered_feature_flag, R_set, C_set, K, nCam=i)

            for k in range(0, i + 1):
                idx_X_pts = np.where(X_found[:, 0] & filtered_feature_flag[:, k])
                x = np.hstack((feature_x[idx_X_pts, k].reshape(-1, 1), feature_y[idx_X_pts, k].reshape(-1, 1)))
                X = X_all[idx_X_pts]
                BundAdj_error = reprojectionErrorPnP(X, x, K, R_set_[k], C_set_[k])
                print("########Error after Bundle Adjustment: ", BundAdj_error)

            print("############Registered camera: ", i + 1, "############################")

    X_found[X_all[:, 2] < 0] = 0
    print("#############DONE###################")

    feature_idx = np.where(X_found[:, 0])
    X = X_all[feature_idx]

    # Plotting results
    plot_2d_trajectory(X, C_set, R_set, output_dir=Output)
    plot_3d_trajectory(X, C_set, output_dir=Output)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X_all)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
