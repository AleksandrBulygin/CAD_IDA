#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>

#define N 12
#define M 25
#define num_train 100
#define num_test 50
#define T 50

float W1[4][N], Wout[N][2], a1[N], d_W1[4][N], d_Wout[N][2], b1[N], d_b1[N],
        bout[2], d_bout[2], z1[N], zout[2],
        y_pred[2], delta_out[2], delta1[N], e1[N], z2[N], b2[N], W2[N][N],
        a2[N], e2[N], delta2[N], d_b2[N], d_W2[N][N],
        **W1_tr, **W2_tr, **Wout_tr, b1_tr[N], b2_tr[N], bout_tr[2],
        y_dif[2], a2_dif[N], a1_dif[N];
int i, j, W1_int[4][N], W2_int[N][N], Wout_int[N][2], b1_int[N], b2_int[N], bout_int[2], x1_int[M], x2_int[M], x3_int[M], x4_int[M];
// Формирование обучающей и тестовой выборок данных
float x1[] =
        {0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 2.5, 0, 1, 1, 2, 2,
         3, 0, 1, 2, 3, 0, 3};
float x2[] =
        {0.0001, 0.0006, 0.3, 0.4, 0.04, 0.2, 0.004, 0.0006, 0.001, 0.002,
         0.0006, 0.0009, 0.00002,
         0.005, 0.001, 0.0005, 0.002, 0.07, 0.2, 0.0007, 0.002, 0.00002,
         0.002, 0.05, 0.1};
float x3[] =
        {1.11, 0, 2.22, 1.12, 3.33, 2.23, 4.44, 3.33, 5.56, 4.44, 6.67, 5.56,
         0, 2.22, 0,
         4.44, 2.22, 6.67, 4.45, 4.44, 6.67, 0, 2.22, 6.67, 0.002};
float x4[] =
        {0.5, 0, 1, 0.5, 1.5, 1, 2, 1.5, 2.5, 2, 3, 2.5, 0, 1, 0, 2, 1, 3, 2,
         2, 3, 0, 1, 3, 0};
float P[] =
        {0.60, 0.04, 0.03, 0.38, 0, 0.45, 0.17, 0.87, 0.08, 0.54, 0.13, 0.17,
         0.83, 0.50, 0.53,
         0.58, 0.33, 0.64, 0.72, 0.04, 0.92, 0.98, 0.88, 1};
float I[] =
        {0.45, 0.17, 0.17, 0, 0.06, 0.05, 0.21, 0.67, 0.41, 0.59, 0.66, 0.72,
         0.37, 0.4,
         0.56, 0.5, 0.12, 0.1, 0.61, 0.83, 1, 0.72, 0.32, 0.3};
float x1_test[] = {0.1, 0.6, 1.5, 3.0, 4.0, 2.0, 1.8, 0.9, 2.7};
float x2_test[] = {0.0002, 0.02, 0.1, 0.0003, 0.0002, 0.0006, 0.3, 0.0001,
                   0.006};
float x3_test[] = {1.33, 0.22, 6.66, 8.89, 6.67, 1.56, 2.01, 4, 2.22};
float x4_test[] = {0.6, 0.1, 3.0, 4.0, 3.0, 0.7, 0.9, 1.8, 1.0};
float P_test[] = {0.45, 0.04, 0.91, 0.65, 0.15, 0.3, 0.46, 0.03, 0.97};
float I_test[] = {0.5, 0.1, 0.87, 0.32, 0.67, 0.08, 0.04, 0.59, 0.33};
int main() {
    // файлы для записи массивов порогов и весов после обучения
    FILE *W1_f = NULL;
    W1_f = fopen("W1_int.txt", "w");
    FILE *W2_f = NULL;
    W2_f = fopen("W2_int.txt", "w");
    FILE *Wout_f = NULL;
    Wout_f = fopen("Wout_int.txt", "w");
    FILE *b1_f = NULL;
    b1_f = fopen("b1_int.txt", "w");
    FILE *b2_f = NULL;
    b2_f = fopen("b2_int.txt", "w");
    FILE *bout_f = NULL;
    bout_f = fopen("bout_int.txt", "w");
    FILE *W1_ff = NULL;
    W1_ff = fopen("W1_float.txt", "w");
    FILE *W2_ff = NULL;
    W2_ff = fopen("W2_float.txt", "w");
    FILE *Wout_ff = NULL;
    Wout_ff = fopen("Wout_float.txt", "w");
    FILE *b1_ff = NULL;
    b1_ff = fopen("b1_float.txt", "w");
    FILE *b2_ff = NULL;
    b2_ff = fopen("b2_float.txt", "w");
    FILE *bout_ff = NULL;
    bout_ff = fopen("bout_float.txt", "w");
    W1_tr = (float **) malloc(4 * sizeof(float *));
    for (i = 0 ; i < 4 ; i++) W1_tr[i] = (float *) malloc(N * sizeof(float
        *));
    W2_tr = (float **) malloc(N * sizeof(float *));
    for (i = 0 ; i < N ; i++) W2_tr[i] = (float *) malloc(N * sizeof(float
        *));
    Wout_tr = (float **) malloc(N * sizeof(float *));
    for (i = 0 ; i < N ; i++) Wout_tr[i] = (float *) malloc(2 * sizeof(float
        *));
// функция обучения нейросети
    nn_train(x1, x2, x3, x4, P, I, W1, Wout, a1, d_W1, d_Wout, b1, d_b1,
             bout, d_bout, z1, zout, y_pred, delta_out,
             delta1, e1, z2, b2, W2, a2, e2, delta2, d_b2, d_W2, W1_tr,
             W2_tr, Wout_tr, b1_tr, b2_tr, bout_tr, y_dif,
             a2_dif, a1_dif, x1_test, x2_test, x3_test, x4_test, P_test,
             I_test);
    // перевод всех значений в формат Q16.15, запись массивов параметров нейросети в файлы
    for (i = 0 ; i < 4 ; i++) {
        for (j = 0 ; j < N ; j++) {
            W1_int[i][j] = W1_tr[i][j] * 32768;
            fprintf(W1_f, " %d", W1_int[i][j]);
            fprintf(W1_ff, " %f", W1_tr[i][j]);
        }
    }
    for (i = 0 ; i < N ; i++) {
        b1_int[i] = b1_tr[i] * 32768;
        fprintf(b1_f, " %d", b1_int[i]);
        fprintf(b1_ff, " %f", b1_tr[i]);
    }
    for (i = 0 ; i < N ; i++) {
        for (j = 0 ; j < N ; j++) {
            W2_int[i][j] = W2_tr[i][j] * 32768;
            fprintf(W2_f, " %d", W2_int[i][j]);
            fprintf(W2_ff, " %f", W2_tr[i][j]);
        }
    }
    for (i = 0 ; i < N ; i++) {
        b2_int[i] = b2_tr[i] * 32768;
        fprintf(b2_f, " %d", b2_int[i]);
        fprintf(b2_ff, " %f", b2_tr[i]);
    }
    for (i = 0 ; i < N ; i++) {
        for (j = 0 ; j < 2 ; j++) {
            Wout_int[i][j] = Wout_tr[i][j] * 32768;
            fprintf(Wout_f, " %d", Wout_int[i][j]);
            fprintf(Wout_ff, " %f", Wout_tr[i][j]);
        }
    }
    for (i = 0 ; i < 2 ; i++) {
        bout_int[i] = bout_tr[i] * 32768;
        fprintf(bout_f, " %d", bout_int[i]);
        fprintf(bout_ff, " %f", bout_tr[i]);
    }
    // формат Q23.7
    float y_true[2][M];
    for (i = 0 ; i < M ; i++) {
        x1_int[i] = x1[i] * 128;
        x2_int[i] = x2[i] * 128;
        x3_int[i] = x3[i] * 128;
        x4_int[i] = x4[i] * 128;
        y_true[0][i] = P[i];
        y_true[1][i] = I[i];
    }
    fclose(W1_f);
    fclose(W2_f);
    fclose(Wout_f);
    fclose(b1_f);
    fclose(b2_f);
    fclose(bout_f);
    fclose(W1_ff);
    fclose(W2_ff);
    fclose(Wout_ff);
    fclose(b1_ff);
    fclose(b2_ff);
    fclose(bout_ff);
    return 0;
}
