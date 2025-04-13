#include "parallel.h"
#include <omp.h>

// 定义全局变量
double total_comm_time = 0.0; // 通信时间
int total_comm_count = 0;     // 通信次数
double start_time, end_time;
int totalcount = 0;  
/*// 发送矩阵列数据
void sendMatrixColumn(const MatrixXd& src_matrix, int src_col, 
                      std::vector<double>& send_buffer, 
                      int target_rank, int tag) {
    int rows = src_matrix.rows();
    send_buffer.resize(rows);
    for (int i = 0; i < rows; i++) {
        send_buffer[i] = src_matrix(i, src_col);
    }
    MPI_Send(send_buffer.data(), rows, MPI_DOUBLE, target_rank, tag, MPI_COMM_WORLD);
}

void recvMatrixColumn(std::vector<double>& recv_buffer, 
                      int src_rank, int tag) {
    int rows = recv_buffer.size();
    MPI_Recv(recv_buffer.data(), rows, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}*/

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
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    const int rows = matrix.rows();
    const int cols = matrix.cols();

    vector<double> send_left_0(rows), send_left_1(rows);
    vector<double> send_right_0(rows), send_right_1(rows);
    vector<double> recv_left_0(rows), recv_left_1(rows);
    vector<double> recv_right_0(rows), recv_right_1(rows);

    // 填充数据
    for (int i = 0; i < rows; i++) {
        send_left_0[i] = matrix(i, 2);
        send_left_1[i] = matrix(i, 3);
        send_right_0[i] = matrix(i, cols-4);
        send_right_1[i] = matrix(i, cols-3);
    }

    MPI_Request requests[8];
    int req_count = 0;

    // 确定左右邻居
    int left_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == num_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // 左邻居通信
    if (left_rank != MPI_PROC_NULL) {
        sendMatrixColumnWithSafety(matrix, 2, send_left_0, left_rank, 0);
        sendMatrixColumnWithSafety(matrix, 3, send_left_1, left_rank, 1);
        recvMatrixColumnWithSafety(recv_left_0, left_rank, 2);
        recvMatrixColumnWithSafety(recv_left_1, left_rank, 3);
    }

    // 右邻居通信
    if (right_rank != MPI_PROC_NULL) {
        sendMatrixColumnWithSafety(matrix, cols - 4, send_right_0, right_rank, 2);
        sendMatrixColumnWithSafety(matrix, cols - 3, send_right_1, right_rank, 3);
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
}

/*
// 稳定版列交换（阻塞通信）
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs) {
    const int rows = matrix.rows();
    const int nSend = (rank == 0 || rank == num_procs-1) ? 2 : 4;

    std::vector<std::vector<double>> send_buffers(nSend, std::vector<double>(rows));
    std::vector<std::vector<double>> recv_buffers(nSend, std::vector<double>(rows));

    // 0号进程 → 右
    if (rank == 0) {
        sendMatrixColumn(matrix, matrix.cols()-4, send_buffers[0], rank+1, 0);
        sendMatrixColumn(matrix, matrix.cols()-3, send_buffers[1], rank+1, 1);

        recvMatrixColumn(recv_buffers[0], rank+1, 2);
        recvMatrixColumn(recv_buffers[1], rank+1, 3);
    }
    // 最后一个进程 → 左
    else if (rank == num_procs-1) {
        sendMatrixColumn(matrix, 2, send_buffers[0], rank-1, 2);
        sendMatrixColumn(matrix, 3, send_buffers[1], rank-1, 3);

        recvMatrixColumn(recv_buffers[0], rank-1, 0);
        recvMatrixColumn(recv_buffers[1], rank-1, 1);
    }
    // 中间进程 ↔ 左右
    else {
        // 发送到左
        sendMatrixColumn(matrix, 2, send_buffers[0], rank-1, 2);
        sendMatrixColumn(matrix, 3, send_buffers[1], rank-1, 3);

        // 发送到右
        sendMatrixColumn(matrix, matrix.cols()-4, send_buffers[2], rank+1, 0);
        sendMatrixColumn(matrix, matrix.cols()-3, send_buffers[3], rank+1, 1);

        // 接收来自左
        recvMatrixColumn(recv_buffers[0], rank-1, 0);
        recvMatrixColumn(recv_buffers[1], rank-1, 1);

        // 接收来自右
        recvMatrixColumn(recv_buffers[2], rank+1, 2);
        recvMatrixColumn(recv_buffers[3], rank+1, 3);
    }

    // 拷贝recv_buffer到对应位置
    if (rank == 0) {
        for (int i = 0; i < rows; i++) {
            matrix(i, matrix.cols()-2) = recv_buffers[0][i];
            matrix(i, matrix.cols()-1) = recv_buffers[1][i];
        }
    }
    else if (rank == num_procs-1) {
        for (int i = 0; i < rows; i++) {
            matrix(i, 0) = recv_buffers[0][i];
            matrix(i, 1) = recv_buffers[1][i];
        }
    }
    else {
        for (int i = 0; i < rows; i++) {
            matrix(i, 0) = recv_buffers[0][i];
            matrix(i, 1) = recv_buffers[1][i];
            matrix(i, matrix.cols()-2) = recv_buffers[2][i];
            matrix(i, matrix.cols()-1) = recv_buffers[3][i];
        }
    }

    // 全体同步
    MPI_Barrier(MPI_COMM_WORLD);
}*/

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

// 并行共轭梯度（CG）算法实现
// 输入：
// A - 系数矩阵（稀疏格式）
// b - 右端项向量
// x - 初始解向量，结果将存储在此
// epsilon - 收敛精度
// max_iter - 最大迭代次数
// rank - 当前进程的标识符（MPI）
// num_procs - 总进程数量（MPI）

void CG_parallel(Equation& equ, Mesh mesh, VectorXd& b, VectorXd& x, double epsilon, 
                int max_iter, int rank, int num_procs, double& r0) {
    int n = equ.A.rows();
    SparseMatrix<double> A = equ.A;
     
    // 计算初始残差
    VectorXd r = b - A * x;//矩阵向量乘 n1
    MPI_Barrier(MPI_COMM_WORLD);
    MatrixXd r_field(mesh.ny+2, mesh.nx+2), x_field(mesh.ny+2, mesh.nx+2);
    //交换矩阵重叠区域并计算
    vectorToMatrix(r, r_field, mesh);
    vectorToMatrix(x, x_field, mesh);
    MPI_Barrier(MPI_COMM_WORLD);
    exchangeColumns(x_field, rank, num_procs);
    MPI_Barrier(MPI_COMM_WORLD);
    //修正重叠单元
    Parallel_correction2(mesh, equ, r_field, x_field);
    MPI_Barrier(MPI_COMM_WORLD);
    //写回向量
    matrixToVector(r_field, r, mesh);
    MPI_Barrier(MPI_COMM_WORLD);
    VectorXd p = r;         
    VectorXd Ap(n);
    
    // 计算初始残差和基准残差
    double r_norm = r.squaredNorm();
    double b_norm = b.squaredNorm();
    
    double local_b_norm = b_norm;
    double global_b_norm;
    // 使用 MPI_Allreduce 来将各个进程的 b_norm 求最大值（或总和等），确保每个进程能够看到全局的 b_norm
MPI_Allreduce(&local_b_norm, &global_b_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// 检查全局的 b_norm
if (global_b_norm < 1e-13) {
    x.setZero();
    r0 = 0.0;
    if (rank==0)
    {
       cout<<"全局b_norm小于1e-13，直接返回"<<endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    return;  // 一旦判断为小于阈值，直接在所有进程处执行return
}
    // 使用绝对残差判据
    double tol = epsilon * epsilon; // 直接使用给定的epsilon作为绝对收敛判据
    
    double global_r_norm;
    //全局规约残差
    MPI_Allreduce(&r_norm, &global_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     r0 = sqrt(global_r_norm);  // 更新初始r0
    int iter = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    while (iter < max_iter) {
        // 计算 Ap
        Ap = A * p;
        MatrixXd p_field(mesh.ny+2, mesh.nx+2), Ap_field(mesh.ny+2, mesh.nx+2);
        vectorToMatrix(p, p_field, mesh);
        vectorToMatrix(Ap, Ap_field, mesh);
        MPI_Barrier(MPI_COMM_WORLD);
        exchangeColumns(p_field, rank, num_procs);
        MPI_Barrier(MPI_COMM_WORLD);
        Parallel_correction(mesh, equ, Ap_field, p_field);
        MPI_Barrier(MPI_COMM_WORLD);
        matrixToVector(Ap_field, Ap, mesh);
        MPI_Barrier(MPI_COMM_WORLD);
        // 计算步长
        double local_dot_p_Ap = p.dot(Ap);
        double global_dot_p_Ap;
        MPI_Allreduce(&local_dot_p_Ap, &global_dot_p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double alpha = global_r_norm / global_dot_p_Ap;
        MPI_Barrier(MPI_COMM_WORLD);
        // 更新解和残差
        x += alpha * p;
        r -= alpha * Ap;

        // 计算新残差范数
        double new_r_norm = r.squaredNorm();
        double global_new_r_norm;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&new_r_norm, &global_new_r_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // 更新r0为当前全局残差
        MPI_Barrier(MPI_COMM_WORLD);
        // 使用绝对残差判断收敛性
       

        // 更新搜索方向
        double beta = global_new_r_norm / global_r_norm;
        p = r + beta * p;
        global_r_norm = global_new_r_norm;

        // 保存当前残差
        r0 = sqrt(global_r_norm);

        /*if(rank == 0 && iter % 5 == 0) {
            std::cout << "Iteration " << iter 
                     << " Absolute residual: " << sqrt(global_new_r_norm) 
                     << std::endl;
        }*/

        iter++;
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // 确保最终r0同步
   // MPI_Bcast(&r0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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