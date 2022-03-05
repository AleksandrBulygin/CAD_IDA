#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 12
#define M 24
#define num_train 100
#define num_test 50

typedef struct FP_struct{
    int P;
    int I;
} FP_out;

FP_out nn_opt(int x1, int x2, int x3, int x4, int W1_tr[4][N], int W2_tr[N][N], int Wout_tr[N][2],
              int b1_tr[N], int b2_tr[N], int bout_tr[2], short tabl_sigma[1024]);

float x1[] = { 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 2.5, 0, 1, 1, 2,
               2, 3, 0, 1, 2, 3, 0, 3 };
float x2[] = { 0.0001, 0.0006, 0.3, 0.4, 0.04, 0.2, 0.004, 0.0006, 0.001, 0.002,
               0.0006, 0.0009, 0.00002, 0.005, 0.001, 0.0005, 0.002, 0.07, 0.2, 0.0007,
               0.002, 0.00002, 0.002, 0.05, 0.1 };
float x3[] = { 1.11, 0, 2.22, 1.12, 3.33, 2.23, 4.44, 3.33, 5.56, 4.44, 6.67,
               5.56, 0, 2.22, 0, 4.44, 2.22, 6.67, 4.45, 4.44, 6.67, 0, 2.22, 6.67,
               0.002 };
float x4[] = { 0.5, 0, 1, 0.5, 1.5, 1, 2, 1.5, 2.5, 2, 3, 2.5, 0, 1, 0, 2, 1, 3,
               2, 2, 3, 0, 1, 3, 0 };
float P[] = { 0.59, 0.03, 0.01, 0.37, 0.03, 0.45, 0.16, 0.9, 0.08, 0.55, 0.12,
              0.17, 0.85, 0.50, 0.52, 0.56, 0.34, 0.64, 0.73, 0.04, 0.92, 0.97, 0.84,
              0.99 };
float I[] = { 0.44, 0.17, 0.15, 0.02, 0.09, 0.05, 0.2, 0.64, 0.4, 0.59, 0.66,
              0.71, 0.36, 0.39, 0.55, 0.49, 0.07, 0.1, 0.6, 0.83, 0.99, 0.71, 0.33, 0.29 };


long long z2[N];
int x1_int[M], x2_int[M], x3_int[M], x4_int[M], W1_r[4][N], W2_r[N][N], Wout_r[N][2], b1_r[N], b2_r[N],
        bout_r[2];
int i, j;
short tabl_sigma[1024];
int main() {
    FP_out y_out[M];
    FILE *Read_x1 = fopen("x1.txt", "r");
    for (i = 0; i < num_train; i++)
        fscanf(Read_x1, "%d", &x1[i]);
    fclose(Read_x1); FILE *Read_x2 = fopen("x2.txt", "r");
    for (i = 0; i < num_train; i++)
        fscanf(Read_x2, "%d", &x2[i]);
    fclose(Read_x2);
    FILE *Read_x3 = fopen("x3.txt", "r");
    for (i = 0; i < num_train; i++)
        fscanf(Read_x3, "%d", &x3[i]);
    fclose(Read_x1);
    FILE *Read_x4 = fopen("x4.txt", "r");
    for (i = 0; i < num_train; i++)
        fscanf(Read_x4, "%d", &x4[i]);
    fclose(Read_x4);
    FILE *Read_I = fopen("I.txt", "r");
    for (i = 0; i < num_train; i++)
        fscanf(Read_I, "%d", &I[i]);
    fclose(Read_I);
    FILE *Read_P = fopen("P.txt", "r");
    for (i = 0; i < num_train; i++)
        fscanf(Read_P, "%d", &P[i]);
    fclose(Read_P);
    for (i = 0; i < 1024; i++) {
        float x = ((float) i - 512) / 32;
        int index = (i - 512) & 1023;
        float y = 1 / (1 + exp(-x));
        int y_Q15 = y * 32768;
        tabl_sigma[index] = y_Q15;
    }
    float y_true[2][M];
    for (i = 0; i < M; i++) {
        x1_int[i] = x1[i] * 128;
        x2_int[i] = x2[i] * 128;
        x3_int[i] = x3[i] * 128;
        x4_int[i] = x4[i] * 128;
        y_true[0][i] = P[i];
        y_true[1][i] = I[i];
    }
    FILE *Read_W1 = fopen("W1_int.txt", "r");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < N; j++)
            fscanf(Read_W1, "%d", &W1_r[i][j]);
    }
    fclose(Read_W1);
    FILE *Read_W2 = fopen("W2_int.txt", "r");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++)
            fscanf(Read_W2, "%d", &W2_r[i][j]);
    }
    fclose(Read_W2);
    FILE *Read_Wout = fopen("Wout_int.txt", "r");
    for (i = 0; i < N; i++) {
        for (j = 0; j < 2; j++)
            fscanf(Read_Wout, "%d", &Wout_r[i][j]);
    }
    fclose(Read_Wout);
    FILE *Read_b1 = fopen("b1_int.txt", "r");
    for (i = 0; i < N; i++)
        fscanf(Read_b1, "%d", &b1_r[i]);
    fclose(Read_b1);
    FILE *Read_b2 = fopen("b2_int.txt", "r");
    for (i = 0; i < N; i++)
        fscanf(Read_b2, "%d", &b2_r[i]);
    fclose(Read_b2);
    FILE *Read_bout = fopen("bout_int.txt", "r");
    for (i = 0; i < 2; i++)
        fscanf(Read_bout, "%d", &bout_r[i]);
    fclose(Read_bout);
    for (int i = 0; i < M; i++) {
        y_out[i] = nn_opt(x1_int[i], x2_int[i], x3_int[i], x4_int[i], W1_r, W2_r, Wout_r, b1_r, b2_r,
               bout_r,  tabl_sigma);
    }
// check
    for (i = 0; i < M; i++) {
        for (j = 0; j < 2; j++) {
            if (j == 0)
                printf("P_out_opt = %.2f, P_true = %.2f\n",
                       (float) y_out[i].P / 32768, y_true[j][i]);
            else
                printf("I_out_opt = %.2f, I_true = %.2f\n",
                       (float) y_out[i].I / 32768, y_true[j][i]);

        }
    }
    return 0;
}
