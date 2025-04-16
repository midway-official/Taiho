#include "parallel.h"
#include <omp.h>

// 定义全局变量
double total_comm_time = 0.0; // 通信时间
int total_comm_count = 0;     // 通信次数
double start_time, end_time;
int totalcount = 0;  

double computeHash(const vector<double>& data) {
    double hash = 2166136261.0;  // FNV offset basis
    const double prime = 16777619.0;
    for (double val : data) {
        hash = fmod((hash * prime), 1e18);  // 保持数值稳定
        hash += val;
    }
    return hash;
}

// 发送矩阵列数据，并验证哈希值
void sendMatrixColumnWithSafety(const MatrixXd& src_matrix, int src_col, 
    vector<double>& send_buffer, 
    int target_rank, int tag) {
int rows = src_matrix.rows();
send_buffer.resize(rows + 1);  // 多一个位置放 hash

for (int i = 0; i < rows; i++) {
send_buffer[i+1] = src_matrix(i, src_col);
}

double hash_value = computeHash(vector<double>(send_buffer.begin() + 1, send_buffer.end()));
send_buffer[0] = hash_value;

MPI_Request request;
MPI_Isend(send_buffer.data(), rows+1, MPI_DOUBLE, target_rank, tag, MPI_COMM_WORLD, &request);
MPI_Wait(&request, MPI_STATUS_IGNORE);
}
void recvMatrixColumnWithSafety(vector<double>& recv_buffer, int src_rank, int tag) {
    int rows = recv_buffer.size();
    vector<double> full_buffer(rows + 1);  // 接收 hash+数据

    MPI_Request request;
    MPI_Irecv(full_buffer.data(), rows+1, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    double recv_hash = full_buffer[0];
    for (int i = 0; i < rows; i++) {
        recv_buffer[i] = full_buffer[i+1];
    }

    double computed_hash = computeHash(recv_buffer);
    if (abs(computed_hash - recv_hash) > 1e-5) {
        cerr << "数据校验失败！哈希不匹配！" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
/*
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();

    // 初始化缓冲区，先全置 0
    vector<double> send_left_0(rows, 0.0), send_left_1(rows, 0.0);
    vector<double> send_right_0(rows, 0.0), send_right_1(rows, 0.0);
    vector<double> recv_left_0(rows, 0.0), recv_left_1(rows, 0.0);
    vector<double> recv_right_0(rows, 0.0), recv_right_1(rows, 0.0);

    // 填充发送数据
    for (int i = 0; i < rows; i++) {
        send_left_0[i]  = matrix(i, 2);
        send_left_1[i]  = matrix(i, 3);
        send_right_0[i] = matrix(i, cols - 4);
        send_right_1[i] = matrix(i, cols - 3);
    }

    int left_rank  = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // 左邻居通信
    if (left_rank != MPI_PROC_NULL) {
        sendMatrixColumnWithSafety(matrix, 2, send_left_0, left_rank, 0);
        sendMatrixColumnWithSafety(matrix, 3, send_left_1, left_rank, 1);
        // 先清空接收缓冲区，虽然构造时是0，这里显式清一下更安全
        std::fill(recv_left_0.begin(), recv_left_0.end(), 0.0);
        std::fill(recv_left_1.begin(), recv_left_1.end(), 0.0);
        recvMatrixColumnWithSafety(recv_left_0, left_rank, 2);
        recvMatrixColumnWithSafety(recv_left_1, left_rank, 3);
    }

    // 右邻居通信
    if (right_rank != MPI_PROC_NULL) {
        sendMatrixColumnWithSafety(matrix, cols - 4, send_right_0, right_rank, 2);
        sendMatrixColumnWithSafety(matrix, cols - 3, send_right_1, right_rank, 3);
        // 同样清空接收缓冲区
        std::fill(recv_right_0.begin(), recv_right_0.end(), 0.0);
        std::fill(recv_right_1.begin(), recv_right_1.end(), 0.0);
        recvMatrixColumnWithSafety(recv_right_0, right_rank, 0);
        recvMatrixColumnWithSafety(recv_right_1, right_rank, 1);
    }

    // 更新矩阵
    if (left_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, 0) = recv_left_0[i];
            matrix(i, 1) = recv_left_1[i];
        }
    }
    if (right_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, cols - 2) = recv_right_0[i];
            matrix(i, cols - 1) = recv_right_1[i];
        }
    }
}*/


/*
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();

    // 每次发两列，接收两列，打包后统一处理
    vector<double> send_left_buffer(2 * rows, 0.0);
    vector<double> send_right_buffer(2 * rows, 0.0);
    vector<double> recv_left_buffer(2 * rows, 0.0);
    vector<double> recv_right_buffer(2 * rows, 0.0);

    int left_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Request requests[4];
    int req_count = 0;

    // 打包左邻居要发的数据
    if (left_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            send_left_buffer[i] = matrix(i, 2);    // 第2列
            send_left_buffer[i + rows] = matrix(i, 3);  // 第3列
        }
        MPI_Isend(send_left_buffer.data(), 2 * rows, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(recv_left_buffer.data(), 2 * rows, MPI_DOUBLE, left_rank, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // 打包右邻居要发的数据
    if (right_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            send_right_buffer[i] = matrix(i, cols - 4);    // 倒数第4列
            send_right_buffer[i + rows] = matrix(i, cols - 3);  // 倒数第3列
        }
        MPI_Isend(send_right_buffer.data(), 2 * rows, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(recv_right_buffer.data(), 2 * rows, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // 等待所有请求完成
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // 更新矩阵
    if (left_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            // 确保接收到的数据有效，防止 NaN
            if (std::isnan(recv_left_buffer[i])) {
                std::cerr << "Rank " << rank << " detected NaN at left buffer [" << i << "]\n";
                MPI_Abort(MPI_COMM_WORLD, 1);  // 发现 NaN 时终止程序
            }
            matrix(i, 0) = recv_left_buffer[i];  // 更新第0列

            if (std::isnan(recv_left_buffer[i + rows])) {
                std::cerr << "Rank " << rank << " detected NaN at left buffer [" << i + rows << "]\n";
                MPI_Abort(MPI_COMM_WORLD, 1);  // 发现 NaN 时终止程序
            }
            matrix(i, 1) = recv_left_buffer[i + rows];  // 更新第1列
        }
    }

    if (right_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            // 确保接收到的数据有效，防止 NaN
            if (std::isnan(recv_right_buffer[i])) {
                std::cerr << "Rank " << rank << " detected NaN at right buffer [" << i << "]\n";
                MPI_Abort(MPI_COMM_WORLD, 1);  // 发现 NaN 时终止程序
            }
            matrix(i, cols - 2) = recv_right_buffer[i];  // 更新倒数第2列

            if (std::isnan(recv_right_buffer[i + rows])) {
                std::cerr << "Rank " << rank << " detected NaN at right buffer [" << i + rows << "]\n";
                MPI_Abort(MPI_COMM_WORLD, 1);  // 发现 NaN 时终止程序
            }
            matrix(i, cols - 1) = recv_right_buffer[i + rows];  // 更新倒数第1列
        }
    }
}*/
//


//超高性能的列交换函数
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();

    // 确定左右邻居
    int left_rank  = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // 聚合通信：打包4列 {2,3,cols-4,cols-3}
    const int num_cols_per_side = 2;

    // 分配并清零缓冲区
    vector<double> sendbuf_left(rows * num_cols_per_side, 0.0);
    vector<double> sendbuf_right(rows * num_cols_per_side, 0.0);
    vector<double> recvbuf_left(rows * num_cols_per_side, 0.0);
    vector<double> recvbuf_right(rows * num_cols_per_side, 0.0);

    // 填充发送缓冲区
    for (int i = 0; i < rows; i++) {
        sendbuf_left[i * num_cols_per_side + 0] = matrix(i, 2);
        sendbuf_left[i * num_cols_per_side + 1] = matrix(i, 3);

        sendbuf_right[i * num_cols_per_side + 0] = matrix(i, cols - 4);
        sendbuf_right[i * num_cols_per_side + 1] = matrix(i, cols - 3);
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步

    // 建立 persistent communicator topology（cartesian 拓扑）
    MPI_Comm cart_comm;
    int dims[1] = { num_procs };
    int periods[1] = { 0 };
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart_comm);

    // 定义 persistent request
    MPI_Request requests[4];
    MPI_Send_init(sendbuf_left.data(),  rows * num_cols_per_side, MPI_DOUBLE, left_rank,  0, MPI_COMM_WORLD, &requests[0]);
    MPI_Recv_init(recvbuf_left.data(),  rows * num_cols_per_side, MPI_DOUBLE, left_rank,  1, MPI_COMM_WORLD, &requests[1]);
    MPI_Send_init(sendbuf_right.data(), rows * num_cols_per_side, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Recv_init(recvbuf_right.data(), rows * num_cols_per_side, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &requests[3]);

    MPI_Barrier(MPI_COMM_WORLD); // 同步

    // 启动通信
    MPI_Startall(4, requests);

    

    // 等待通信完成
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD); // 同步

    // 更新矩阵边界值
    if (left_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, 0) = recvbuf_left[i * num_cols_per_side + 0];
            matrix(i, 1) = recvbuf_left[i * num_cols_per_side + 1];
        }
    }

    if (right_rank != MPI_PROC_NULL) {
        for (int i = 0; i < rows; i++) {
            matrix(i, cols - 2) = recvbuf_right[i * num_cols_per_side + 0];
            matrix(i, cols - 1) = recvbuf_right[i * num_cols_per_side + 1];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // 同步

    // 释放 persistent request 和 communicator
    for (int i = 0; i < 4; ++i) {
        MPI_Request_free(&requests[i]);
    }
    MPI_Comm_free(&cart_comm);
}


// 从解向量转换为场矩阵
void vectorToMatrix(const VectorXd& x, MatrixXd& phi, const Mesh& mesh) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                int n = mesh.interid(i, j) - 1; // 获取对应的解向量索引
                phi(i, j) = x[n];
            }
        }
    }
}



// 从场矩阵转换为解向量
void matrixToVector(const MatrixXd& phi, VectorXd& x, const Mesh& mesh) {
    for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                int n = mesh.interid(i, j) - 1; // 获取对应的解向量索引
                x[n] = phi(i, j);
            }
        }
    }
}



void Parallel_correction(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2){
for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                if (mesh.bctype(i, j+1) == -3)
                {
                     phi1(i, j)-=equ.A_e(i, j)*phi2(i, j+1);
                }
                 if (mesh.bctype(i, j-1) == -3)
                {
                     phi1(i, j)-=equ.A_w(i, j)*phi2(i, j-1);
                }
                
               
            }
        }
    }
}
void Parallel_correction2(Mesh mesh,Equation equ,MatrixXd &phi1,MatrixXd &phi2){
for (int i = 0; i <= mesh.ny + 1; i++) {
        for (int j = 0; j <= mesh.nx + 1; j++) {
            if (mesh.bctype(i, j) == 0) { // 仅处理内部点
                if (mesh.bctype(i, j+1) == -3)
                {
                     phi1(i, j)+=equ.A_e(i, j)*phi2(i, j+1);
                }
                 if (mesh.bctype(i, j-1) == -3)
                {
                     phi1(i, j)+=equ.A_w(i, j)*phi2(i, j-1);
                }
                
               
            }
        }
    }
}
void CG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon, 
    int max_iter, int rank, int num_procs, double& r0) {
int n = equ.A.rows();
SparseMatrix<double> A = equ.A;

// 计算初始残差
VectorXd r = VectorXd::Zero(n); // 先初始化为0
r = b - A * x; // 矩阵向量乘 n1
MPI_Barrier(MPI_COMM_WORLD);

// 矩阵场变量初始化
MatrixXd r_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);
MatrixXd x_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);

//交换矩阵重叠区域并计算
vectorToMatrix(r, r_field, mesh);
vectorToMatrix(x, x_field, mesh);
MPI_Barrier(MPI_COMM_WORLD);
exchangeColumns(x_field, rank, num_procs);
MPI_Barrier(MPI_COMM_WORLD);
Parallel_correction2(mesh, equ, r_field, x_field);
MPI_Barrier(MPI_COMM_WORLD);
matrixToVector(r_field, r, mesh);
MPI_Barrier(MPI_COMM_WORLD);

VectorXd p = VectorXd::Zero(n);
p = r;
VectorXd Ap = VectorXd::Zero(n);

// 初始残差和基准残差
double r_norm = r.squaredNorm();
double b_norm = b.squaredNorm();
double local_b_norm = b_norm;
double global_b_norm = 0.0;

// 规约 b_norm
MPI_Allreduce(&local_b_norm, &global_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

if (global_b_norm < 1e-13) {
x.setZero();
r0 = 0.0;
if (rank == 0) {
std::cout << "全局b_norm小于1e-13，直接返回" << std::endl;
}
MPI_Barrier(MPI_COMM_WORLD);
return;
}

double tol = epsilon * epsilon;
double global_r_norm = 0.0;
MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
r0 = std::sqrt(global_r_norm);

int iter = 0;
MPI_Barrier(MPI_COMM_WORLD);

while (iter < max_iter) {
Ap.setZero();
Ap = A * p;

MatrixXd p_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);
MatrixXd Ap_field = MatrixXd::Zero(mesh.ny+2, mesh.nx+2);

vectorToMatrix(p, p_field, mesh);
vectorToMatrix(Ap, Ap_field, mesh);
MPI_Barrier(MPI_COMM_WORLD);
exchangeColumns(p_field, rank, num_procs);
MPI_Barrier(MPI_COMM_WORLD);
Parallel_correction(mesh, equ, Ap_field, p_field);
MPI_Barrier(MPI_COMM_WORLD);
matrixToVector(Ap_field, Ap, mesh);
MPI_Barrier(MPI_COMM_WORLD);

double local_dot_p_Ap = p.dot(Ap);
double global_dot_p_Ap = 0.0;
MPI_Allreduce(&local_dot_p_Ap, &global_dot_p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

double alpha = global_r_norm / global_dot_p_Ap;
MPI_Barrier(MPI_COMM_WORLD);

x += alpha * p;
r -= alpha * Ap;

double new_r_norm = r.squaredNorm();
double global_new_r_norm = 0.0;
MPI_Barrier(MPI_COMM_WORLD);
MPI_Allreduce(&new_r_norm, &global_new_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);

double beta = global_new_r_norm / global_r_norm;
p = r + beta * p;
global_r_norm = global_new_r_norm;

r0 = std::sqrt(global_r_norm);
iter++;
MPI_Barrier(MPI_COMM_WORLD);
}

MPI_Barrier(MPI_COMM_WORLD);
}


void BiCGSTAB_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon,
    int max_iter, int rank, int num_procs, double& r0) {
    
int n = equ.A.rows();
SparseMatrix<double> A = equ.A;

VectorXd r = b - A * x;
MatrixXd r_field(mesh.ny+2, mesh.nx+2), x_field(mesh.ny+2, mesh.nx+2);
vectorToMatrix(r, r_field, mesh);
vectorToMatrix(x, x_field, mesh);
exchangeColumns(x_field, rank, num_procs);
Parallel_correction2(mesh, equ, r_field, x_field);
matrixToVector(r_field, r, mesh);

    VectorXd r_tld = r;
    VectorXd p = r, v = VectorXd::Zero(n), s, t;

double rho_old = 1.0, alpha = 1.0, omega = 1.0;
double rho_new, beta;

double r_norm = r.squaredNorm();
double local_b_norm = b.squaredNorm(), global_b_norm;
MPI_Allreduce(&local_b_norm, &global_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

if (global_b_norm < 1e-13) {
x.setZero();
r0 = 0.0;
MPI_Barrier(MPI_COMM_WORLD);
return;
}

double global_r_norm;
MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
r0 = sqrt(global_r_norm);

int iter = 0;
while (iter < max_iter) {
rho_new = r_tld.dot(r);
        

if (iter == 0) {
p = r;
} else {
beta = (rho_new / rho_old) * (alpha / omega);
p = r + beta * (p - omega * v);
}

v = A * p;
MatrixXd p_field(mesh.ny+2, mesh.nx+2), v_field(mesh.ny+2, mesh.nx+2);
vectorToMatrix(p, p_field, mesh);
vectorToMatrix(v, v_field, mesh);
exchangeColumns(p_field, rank, num_procs);
Parallel_correction(mesh, equ, v_field, p_field);
matrixToVector(v_field, v, mesh);

double local_dot = r_tld.dot(v), global_dot;
MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
alpha = rho_new / global_dot;

// s = r - α*v
s = r - alpha * v;

// s 范数（不再提前退出）
double s_norm = s.squaredNorm(), global_s_norm;
MPI_Allreduce(&s_norm, &global_s_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // if (sqrt(global_s_norm) < epsilon) {
        //     x += alpha * p;
        //     break;
        // }

t = A * s;
MatrixXd s_field(mesh.ny+2, mesh.nx+2), t_field(mesh.ny+2, mesh.nx+2);
vectorToMatrix(s, s_field, mesh);
vectorToMatrix(t, t_field, mesh);
exchangeColumns(s_field, rank, num_procs);
Parallel_correction(mesh, equ, t_field, s_field);
matrixToVector(t_field, t, mesh);

double ts_dot = t.dot(s);
double tt_dot = t.dot(t);
double global_ts_dot, global_tt_dot;
MPI_Allreduce(&ts_dot, &global_ts_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(&tt_dot, &global_tt_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
omega = global_ts_dot / global_tt_dot;

x += alpha * p + omega * s;
r = s - omega * t;

r_norm = r.squaredNorm();
MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
r0 = sqrt(global_r_norm);

        // 不再使用残差判断终止
        // if (r0 < epsilon) break;

rho_old = rho_new;
iter++;
    }

MPI_Barrier(MPI_COMM_WORLD);
}