#ifndef PARALLEL_H
#define PARALLEL_H

#include "DNS.h"
#include <mpi.h>
#include <omp.h>
// 声明全局变量（其他文件可访问）
extern double total_comm_time; // 通信时间
extern int total_comm_count,totalcount;   // 通信次数
extern double start_time, end_time;
// 发送矩阵列数据的函数声明
void sendMatrixColumn(const MatrixXd& src_matrix, int src_col, 
                     MatrixXd& dst_matrix, int dst_col, 
                     int target_rank, int tag = 0);

// 接收矩阵列数据的函数声明
void recvMatrixColumn(MatrixXd& dst_matrix, int dst_col,
                     int src_rank, int tag = 0);
//交互网格数据
void exchangeColumns(MatrixXd& matrix, int rank, int num_procs);
void gsstep(Equation& equation, double& l2_norm, Eigen::MatrixXd& phi);
void cgstep(Equation& equation,  double& l2_norm, Eigen::MatrixXd& phi);
void parallelGs_u(Equation& equ_u0, Equation& equ_v0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter,double tor);
void parallelGs_v(Equation& equ_v0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter,double tor);

void parallelGs_p(Equation& equ_p0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter,double tor);

void parallelGs_p_t(Equation& equ_p0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter,double tor);

void parallelCG_p(Equation& equ_p0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter, double tor);

void parallelGs(Equation& equ_p0,double epsilon_uv, double& l2_norm_x, Mesh& mesh, int max_iter, double tor);


void record_communication_time(double start, double end);

void solve2(Equation& equation, double epsilon, double& l2_norm, MatrixXd& phi);

void gsstep2(Equation& equation,  double& l2_norm, Eigen::MatrixXd& phi);
#endif // PARALLEL_H