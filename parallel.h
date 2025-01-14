#ifndef PARALLEL_H
#define PARALLEL_H

#include "DNS.h"
#include <mpi.h>

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
void parallelGs_u(Equation& equ_u0, Equation& equ_v0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter);
void parallelGs_v(Equation& equ_v0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter);

void parallelGs_p(Equation& equ_p0, Equation& equ_u0, double epsilon_uv, double& l2_norm_x, Mesh& mesh, int rank, int num_procs, int max_iter);
#endif // PARALLEL_H