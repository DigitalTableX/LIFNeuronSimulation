/*
2022/8/23 debug end

Program for simulation of random networks with leaky integrate-and-fire neuron

*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#include <time.h>

#include "C\math.c"			//random number (0~1) generation routine
#include "C\time.c"			//return date and time
#include "H\vec.h"  		//definition of dip, nip

//output folder
#define		OUT_FOLDER			"out"

//data label
#define		DLB_MAIN_BASE		"PRG"

#define 	PAI					(atan(1.e0) * 4.e0)
#define		EPS3				1.e-3
#define		EPS6				1.e-6
#define		EPS9				1.e-9
#define		EPS12				1.e-12
#define		BIG30				1.e30
#define		NBG3				1000
#define		NBG6				1000000
#define		NBG9				1000000000
#define		NSA					50
#define		NSB					100
#define		NSC					500
#define		NSD					1000
#define		VER					-2.e6
#define		NER					-20000
#define		VSQ					-30000.e0
#define		NSQ					-30000

#define		N_NN			100			//total number of neurons

#define		MAX_TRIAL		100
#define		MAX_POST		N_NN
#define		MAX_ID_FIRE		N_NN
#define		MAX_ARV			10000
#define		MAX_T_FIRE		10000
#define		MAX_FILE_OPEN	500
#define		TEXT_NUM		200

#define		RANSUU_METHOD		1		//0: ran_fast_RCP
										//1: ran1_RCP
										//2: ran2_RCP
										
//calculation mode
#define		CALMODE				0		//0:network simulation

//debug mode
#define		DEB_PULSE_INP		0		//pulse current

#define		MAX(a,b)		((a) >= (b) ? (a) : (b))
#define		MIN(a,b)		((a) <= (b) ? (a) : (b))
#define		EQ(a,b)			(fabs((a)-(b))<EPS6 ? 1 : 0)
#define		NE(a,b)			(fabs((a)-(b))>EPS6 ? 1 : 0)
#define		LE(a,b)			((a) < (b)+EPS6 ? 1 : 0)
#define		LT(a,b)			((a) < (b)-EPS6 ? 1 : 0)
#define		GE(a,b)			((a) > (b)-EPS6 ? 1 : 0)
#define		GT(a,b)			((a) > (b)+EPS6 ? 1 : 0)
#define		EQS(s1,s2)		(!strcmp((s1),(s2)))

struct SYN_s;

struct SYN_s{
	long num_post;						//post neuron
	long id_post[MAX_POST];				//post neuron ID
	long n_d[MAX_POST];					//synaptic delay
	double w[MAX_POST];					//weight
};

void disp_start_main();
void disp_end_main();
void check_para_main();
void mk_aux_main();
void main_NW();
void c_ave_sd(double v[], long n, double *ave, double *sd);
void output_MNW_text();
void mk_NW();
void mk_Syn();
void perfect_random_NW();
void qs_main();
void final_process();
void output_order(double t_a, double t_b, double *r_bar, long n_jikei_out, char fname[]);
long c_j_tfm(long i, double t, long jb_v[]);
double c_norm(double x, double y);
void c_curr_all(double t, double v_nn[], double s_ampa[]);
void output_QSM(long ns, double t, double v_nn[], double s_ampa[]);
void save_firing_time(double t);
void output_raster(long ns, double t);
void output_raster_each(long f_init, double t);
void output_jikei(long ns, double t, double v_nn[], double s_ampa[]);
void output_jikei_each(long ns, long f_init, double t, double v_nn[], double s_ampa[]);
void synapse_dyn(double s_ampa[]);
void init_QSM(double v[], double s_ampa[]);
void neuron_dyn(double t, double v_nn[], double s_ampa[]);
void send_syn_inp(long i_pre);
void ransuu_init(long i_seed);
double std_gauss_ransuu(void);
void my_fopen(FILE **fp, char fname[], long n_label);
void set_dlb_main_use(int argc, char *argv[]);
float ransuu(void);
void ransuu_init(long i_seed);
void disp_start_end_time(char md[], char c[]);
void disp_err(char c[]);
void output_Con();
void disp_warning();
void add_pulse_inp(double t);
void deb_fun();

double T_step, T_init, T_fin;
long Num_trial;
double Tau_m, E_leak, R_nn, V_thresh, Delta_abs, V_reset;
double G_ampa, Tau_ampa, Weight_ave, Weight_sd, Dlt_min, Dlt_max;
double P_rand_net;
double WN_ave, WN_sd;
long Ransuu_seed_v[MAX_TRIAL];
long Nstep_gamen;
long Fout_jikei;
long Ntr_jikei[NSA];
double T_jikei_min, T_jikei_max;
long Nstep_jikei;
long Id_jikei_sel[NSA];
long Fout_raster;
long Ntr_raster[NSA];
double T_raster_min, T_raster_max;
long Fout_order_jikei;
long Ntr_order_jikei[MAX_TRIAL]; 
double T_order_jikei_min, T_order_jikei_max;
long Fout_order_opt;
double T_order_opt_min, T_order_opt_max;
double T_order_y;

long Id_PLS;
double I_inj_PLS, T_s_PLS, T_f_PLS;

long Con[N_NN][N_NN];				//connection matrix

SYN_s Syn[N_NN];

long Ntr;

double R_bar_v[MAX_TRIAL];
double R_bar_opt, R_bar_sd;

long Id_fire_v[MAX_ID_FIRE];
long Num_id_fire;

double T_fire_m[N_NN][MAX_T_FIRE];
long Num_t_fire[N_NN];

double T_last[N_NN];				//time of last fire

double I_syn[N_NN];					//synaptic current
double Eta[N_NN];					//white noise
double I_nn[N_NN];					//total current

double R_bar_QSM;

double Arv[N_NN][MAX_ARV];
long M_arv;

long Argc_glb;
char Argv_glb[NSA][200];

double T_save_f_min, T_save_f_max;

char Dlb_main[200];
char Dlb_use[TEXT_NUM];

char Time_text[200];

long I_ransuu;

int main(int argc, char *argv[]){
	T_step 		= 0.5e0;
	T_init		= -5.e3;
	T_fin		= 20.e3;
	
	Num_trial		= 8;			//Number of trials

	Tau_m 		= 20.e0;
	E_leak		= -74.e0;
	R_nn 		= 40.e3;
	V_thresh 	= -54.e0;
	Delta_abs	= 2.e0;
	V_reset		= -74.e0;
	
	//synaptic current model
	G_ampa		= 0.5e-6;
	Tau_ampa 	= 5.e0;
	Weight_ave 	= 1.e0;
	Weight_sd  	= 0.e0;
	Dlt_min		= 0.e0;
	Dlt_max		= 0.e0;
	
	P_rand_net	= 0.2e0;

	//white noise
	WN_ave		= 0.52e-3;
	WN_sd		= 0.1*WN_ave;

	nip(Ransuu_seed_v, -1, -1, -Num_trial, NSQ, MAX_TRIAL);
	
	//screen display
	Nstep_gamen	= 10000;

	Fout_jikei  	= 1;
	nip(Ntr_jikei, 0, Num_trial/2, Num_trial-1, NER, NSA);
	T_jikei_min 	= T_init;
	T_jikei_max 	= T_fin;
	Nstep_jikei 	= 1;
	nip(Id_jikei_sel, 6, 9, 10, NER, NSA);

	//raster plot
	Fout_raster		= 1;
	nip(Ntr_raster, 0, Num_trial/2, Num_trial-1, NER, NSA);
	T_raster_min 	= 0.e3;
	T_raster_max 	= T_fin;
	
	//order parameter
	Fout_order_jikei	= 1;
	nip(Ntr_order_jikei, 0, Num_trial/2, Num_trial-1, NER, MAX_TRIAL);
	T_order_jikei_min	= 0.e3;
	T_order_jikei_max	= T_fin-2.e3;
																		
	Fout_order_opt		= 1;
	T_order_opt_min		= 0.e3;
	T_order_opt_max		= T_fin-2.e3;
	
	T_order_y			= 2.e3;
	
	//pulse current
	Id_PLS		= 10;
	I_inj_PLS	= 0.5e-3;
	T_s_PLS 	= 0.e0;
	T_f_PLS 	= 1000.e0;

    set_dlb_main_use(argc, argv);
	check_para_main();
	mk_aux_main(); 
	
	disp_start_main();
    disp_start_end_time("s", Dlb_main);
	
	if (CALMODE == 0) main_NW();
	
	disp_start_end_time("e", Dlb_main);
	disp_end_main();
}

//display output
void disp_start_main(){
	printf("--start of %s--\n", Dlb_main);
	disp_warning();
}

void disp_end_main(){
	disp_warning();	
	printf("--end of %s--\n", Dlb_main);
}

void disp_warning(){
	if (DEB_PULSE_INP == 1) printf("!!!DEB_PULSE_INP:1!!!\n");
}

//parameter check
void check_para_main(){
	if (Num_trial > MAX_TRIAL-5) disp_err("err-CPM-NTR");

	//T_jikei_min(max)
	if (Fout_jikei == 1){
		if (!(GE(T_jikei_min, T_init) && LE(T_jikei_max, T_fin))) disp_err("err-CPM-TJMa"); 
	}
	
	//T_raster_min(max)
	if (Fout_raster == 1){
		if (!(GE(T_raster_min, T_init) && LE(T_raster_max, T_fin))) disp_err("err-CPM-RASa");
	}
	
	//T_order_jikei_min(max)
	if (Fout_order_jikei == 1){
		if (!(GE(T_order_jikei_min-T_order_y, T_init) && LE(T_order_jikei_max+T_order_y, T_fin))) disp_err("err-CPM-ORJKa");
	}
	
	//T_order_opt_min(max)
	if (Fout_order_opt == 1){
		if (!(GE(T_order_opt_min-T_order_y, T_init) && LE(T_order_opt_max+T_order_y, T_fin))) disp_err("err-CPM-OROPa");
	}	
}

void mk_aux_main(){ 
	long i;
	
	//T_save_f_min, T_save_f_max
	if (Fout_order_jikei == 1 && Fout_order_opt == 1){
		T_save_f_min = MIN(T_order_jikei_min, T_order_opt_min) - T_order_y;
		T_save_f_max = MAX(T_order_jikei_max, T_order_opt_max) + T_order_y;
	}
	else if (Fout_order_jikei == 1 && Fout_order_opt == 0){
		T_save_f_min = T_order_jikei_min - T_order_y; T_save_f_max = T_order_jikei_max + T_order_y;
	}
	else if (Fout_order_jikei == 0 && Fout_order_opt == 1){
		T_save_f_min = T_order_opt_min - T_order_y; T_save_f_max = T_order_opt_max + T_order_y;
	}
	else if (Fout_order_jikei == 0 && Fout_order_opt == 0){
		T_save_f_min = VER; T_save_f_max = VER;
	}
	else{
		disp_err("err: aux-main-a");
	}
}

void main_NW(){
	mk_NW();
	
	for (Ntr = 0; Ntr < Num_trial; Ntr++){
		printf("Ntr = %ld\n", Ntr);
		ransuu_init(Ransuu_seed_v[Ntr]);
		qs_main();
		R_bar_v[Ntr] = R_bar_QSM;
	}
	c_ave_sd(R_bar_v, Num_trial, &R_bar_opt, &R_bar_sd);
	output_MNW_text();
}

//ave, sd
void c_ave_sd(double v[], long n, double *ave, double *sd){
	long i;
	double sum1, sum2, wk1;
	
	sum1 = 0.e0; sum2 = 0.e0;
	for (i = 0; i < n; i++){
		sum1 += v[i]; 
		sum2 += v[i] * v[i];
	}
	*ave = sum1 / n;
	wk1 = sum2/n - (*ave) * (*ave);
	if (wk1 < 0.e0) wk1 = 0.e0;
	*sd = sqrt(wk1);
}

//output MNW
void output_MNW_text(){
	long i;
	FILE *fp;
	my_fopen(&fp, "R_bar_v", 1);
	for (i = 0; i < Num_trial; i++){
		fprintf(fp, "R_bar_v[%ld] = %.7le\n", i, R_bar_v[i]);
	}
	
	fprintf(fp, "\n"); 
	fprintf(fp, "R_bar_opt = %.7le\n", R_bar_opt);
	fprintf(fp, "R_bar_sd = %.7le\n", R_bar_sd);
	fclose(fp);
}

//create network
void mk_NW(){
	perfect_random_NW();
	output_Con();
	mk_Syn();
}

void output_Con(){
	FILE *fp;
	long i, j;
	
	my_fopen(&fp, "Con", 1);
	fprintf(fp, "ID_of_post%s, ID_of_pre%s\n", Dlb_main, Dlb_main);
	
	for (i = 0; i < N_NN; i++) for (j = 0; j < N_NN; j++){
		if (Con[i][j] == 1){ fprintf(fp, "%ld, %ld\n", i, j); }
	}
	
	fclose(fp); 
}

void mk_Syn(){
	long i, j, k;
	double d;
	
	for (i = 0; i < N_NN; i++) Syn[i].num_post = 0;
	
	for (i = 0; i < N_NN; i++) for (j = 0; j < N_NN; j++){	
		if (Con[i][j] == 1){
			if (Syn[j].num_post == MAX_POST) disp_err("err: mk-Syn-b");
			k = Syn[j].num_post;
			
			Syn[j].id_post[k] = i;
			
			d = Dlt_min + (Dlt_max - Dlt_min) * ransuu();	
			Syn[j].n_d[k] = MAX(0, d / T_step);
			if (Syn[j].n_d[k] > MAX_ARV-5) disp_err("err: mk-Syn-a"); 
			
			Syn[j].w[k] = Weight_ave + Weight_sd * std_gauss_ransuu();	
			(Syn[j].num_post)++;
		}
	}
}

void perfect_random_NW(){
	long i, j;
	double u;
	
	for (i = 0; i < N_NN; i++){
		for (j = 0; j < N_NN; j++){
			if (i == j) continue;

			u = ransuu();
			if (u < P_rand_net) Con[i][j] = 1; 
			else Con[i][j] = 0;
		}
	}
}

//network simulation
void qs_main(){
	double t;
	double v_nn[N_NN], s_ampa[N_NN];
	long ns;
	
	init_QSM(v_nn, s_ampa);
	
	for (ns = 0;;ns++){
		t = ns * T_step + T_init;
		if (ns % Nstep_gamen == 0) printf("Ntr = %ld, t = %.3lf(sec)\n", Ntr, t/1000.e0);
		
		c_curr_all(t, v_nn, s_ampa);
		neuron_dyn(t, v_nn, s_ampa);
		synapse_dyn(s_ampa);
		output_QSM(ns, t, v_nn, s_ampa);
		
		if (EQ(t, T_fin)){ final_process(); return; }
	}
}

void final_process(){
	double dum;
	static long c;
	char text1[200]; 
	if (Ntr == 0) c = 0;

	if (Fout_order_jikei == 1 && Ntr == Ntr_order_jikei[c]){ 
		sprintf(text1, "r_order#Ntr=%ld", Ntr); 
		output_order(T_order_jikei_min, T_order_jikei_max, &dum, 1, text1);
		c++;
	} 
	
	if (Fout_order_opt == 1){ 
		output_order(T_order_opt_min, T_order_opt_max, &R_bar_QSM, 0, "dum");
	} 
}

//order parameter
void output_order(double t_a, double t_b, double *r_bar, long n_jikei_out, char fname[]){
	long i, j, n;
	long jb_v[N_NN];
	double r_order;
	double t, tf, tf_1, phi, sum_re, sum_im, sum_od;
	double re[N_NN], im[N_NN];
	FILE *fp;
	
	if (n_jikei_out == 1){ my_fopen(&fp, fname, 1); fprintf(fp, "t, t(sec), r_order#Ntr=%ld%s\n", Ntr, Dlb_main); }
	for (i = 0; i < N_NN; i++) jb_v[i] = 0;
	
	sum_od = 0.e0;
	for (n = 0; ;n++){
		t = t_a + T_step * n;
		
		for (i = 0; i < N_NN; i++){
			j = c_j_tfm(i, t, jb_v);
			
			tf = T_fire_m[i][j]; tf_1 = T_fire_m[i][j+1];
			
			phi = 2.e0 * PAI * (t - tf) / (tf_1 - tf);	
			re[i] = cos(phi); im[i] = sin(phi);	
		}
		
		sum_re = 0.e0; sum_im = 0.e0;										
		for (i = 0; i < N_NN; i++){ sum_re += re[i]; sum_im += im[i]; }		
		r_order = c_norm(sum_re, sum_im) / N_NN;							
		
		if (n_jikei_out == 1) fprintf(fp, "%.7le, %.7le, %.7le\n", t, t/1000.e0, r_order);
		
		sum_od = sum_od + r_order * T_step;	
		if (GE(t, t_b)) break;
	}
	*r_bar = sum_od / (t_b - t_a);
	
	if (n_jikei_out == 1) fclose(fp); 
}

long c_j_tfm(long i, double t, long jb_v[]){
	long j;
	for (j = jb_v[i]; j < Num_t_fire[i]-1; j++){
		if ( GE(t, T_fire_m[i][j]) && LT(t, T_fire_m[i][j+1]) ){
			jb_v[i] = j; 
			return j;
		}
	}
	printf("i = %ld, t = %.3lf in c-j-tfm\n", i, t);
	disp_err("err: c-j-tfm");
	return 0;
}

//calculate norm
double c_norm(double x, double y){
	double wk1;
	wk1 = sqrt(x * x + y * y);
	return wk1;
}

//calculate current
void c_curr_all(double t, double v_nn[], double s_ampa[]){
	long i;
	
	for (i = 0; i < N_NN; i++){
		I_syn[i] = G_ampa * s_ampa[i] * v_nn[i];
		Eta[i] = WN_ave + WN_sd * std_gauss_ransuu();
		I_nn[i] = Eta[i] - I_syn[i];
	}
	
	if (DEB_PULSE_INP == 1) add_pulse_inp(t);
}

//add pulse current
void add_pulse_inp(double t){
	if (GE(t, T_s_PLS) && LT(t, T_f_PLS)) I_nn[Id_PLS] += I_inj_PLS;
}

//output
void output_QSM(long ns, double t, double v_nn[], double s_ampa[]){
	if (Fout_jikei == 1) output_jikei(ns, t, v_nn, s_ampa);
	if (Fout_raster == 1) output_raster(ns, t);
	if (GE(t, T_save_f_min) && LE(t, T_save_f_max)) save_firing_time(t);
}

void save_firing_time(double t){
	long i, k;
	
	for (i = 0; i < Num_id_fire; i++){
		k = Id_fire_v[i];
		
		T_fire_m[k][Num_t_fire[k]] = t;
		(Num_t_fire[k])++;
		if (Num_t_fire[k] > MAX_T_FIRE-5) disp_err("err: save-firing-time-a"); 
	}
}

//output raster plot
void output_raster(long ns, double t){
	static long c;
	static long f_init;
	
	if (Ntr == 0 && ns == 0) c = 0;
	
	if (Ntr == Ntr_raster[c] && GE(t, T_raster_min) && LE(t, T_raster_max)){
		if (EQ(t, T_raster_min)){ f_init = 1; }
		else if (EQ(t, T_raster_max)){ f_init = -1; c++; }
		else{ f_init = 0; }
		
		output_raster_each(f_init, t);
	}
}

void output_raster_each(long f_init, double t){
	long i;
	static FILE *fp;
	char text1[200];
    
	if (f_init == 1){ 
		sprintf(text1, "raster#Ntr=%ld", Ntr); my_fopen(&fp, text1, 1); 
		fprintf(fp, "t, t(sec), ID_of_neuron(Ntr=%ld)%s\n", Ntr, Dlb_main);
	}
	
	if (f_init == -1){ fclose(fp); return; }
	
	for (i = 0; i < Num_id_fire; i++){
		fprintf(fp, "%.7le, %.7le, %ld\n", t, t/1000.e0, Id_fire_v[i]);
	}
}

//output time series data
void output_jikei(long ns, double t, double v_nn[], double s_ampa[]){
	static long c;
	static long f_init;
	
	if (Ntr == 0 && ns == 0) c = 0;
	
	if (Ntr == Ntr_jikei[c] && GE(t, T_jikei_min) && LE(t, T_jikei_max)){
		if (EQ(t, T_jikei_min)){ f_init = 1; }
		else if (EQ(t, T_jikei_max)){ f_init = -1; c++; }
		else{ f_init = 0; }
		
		output_jikei_each(ns, f_init, t, v_nn, s_ampa);
	}
}

void output_jikei_each(long ns, long f_init, double t, double v_nn[], double s_ampa[]){
	long i, id;
	static FILE *fp;
	char text1[200];
    
	if (f_init == 1){ 
		sprintf(text1, "state_jikei#Ntr=%ld", Ntr); my_fopen(&fp, text1, 1);
		
		fprintf(fp, "t, t(sec), ");
		for (i = 0; Id_jikei_sel[i] != NER; i++){
			id = Id_jikei_sel[i]; 
			fprintf(fp, "v_nn[%ld], s_ampa[%ld], I_syn[%ld], Eta[%ld], I_nn[%ld]", id, id, id, id, id);
			if (Id_jikei_sel[i+1] != NER) fprintf(fp, ", "); else fprintf(fp, "\n"); 
		}
	}
	
	if (f_init == -1){ fclose(fp); return; }
	if (ns % Nstep_jikei != 0) return;
	
	fprintf(fp, "%.7le, %.7le, ", t, t/1000.e0);
	for (i = 0; Id_jikei_sel[i] != NER; i++){
		id = Id_jikei_sel[i];
		fprintf(fp, "%.7le, %.7le, %.7le, %.7le, %.7le, ", v_nn[id], s_ampa[id], I_syn[id], Eta[id], I_nn[id]);
	}
	fprintf(fp, "\n");
}

//synaptic dynamics
void synapse_dyn(double s_ampa[]){
	long i;
	
	for (i = 0; i < N_NN; i++){
		if (Arv[i][M_arv] > 0.e0){
			s_ampa[i] += Arv[i][M_arv];
			Arv[i][M_arv] = 0.e0;
		}
		s_ampa[i] -= s_ampa[i] / Tau_ampa * T_step;	
	}
	M_arv = (M_arv+1)%MAX_ARV;
}

void init_QSM(double v_nn[], double s_ampa[]){
	long i, j;
	//v_nn, s_ampa
	for (i = 0; i < N_NN; i++){ v_nn[i] = V_reset; s_ampa[i] = 0.e0; }
	
	//Num_t_fire
	for (i = 0; i < N_NN; i++) Num_t_fire[i] = 0;
	
	//T_last
	for (i = 0; i < N_NN; i++) T_last[i] = -BIG30;
	
	//I_syn, Eta, I_nn
	for (i = 0; i < N_NN; i++){ I_syn[i] = 0.e0; Eta[i] = 0.e0; I_nn[i] = 0.e0; } 
	
	//R_bar_QSM;
	R_bar_QSM = -1.e0;
	
	//Arv, M_arv
	for (i = 0; i < N_NN; i++){
		for (j = 0; j < MAX_ARV; j++){
			Arv[i][j] = 0.e0;
		}
	}
	M_arv = 0;
}

//neuron dynamics
void neuron_dyn(double t, double v_nn[], double s_ampa[]){
	long i;
	double v_dt;
	
	Num_id_fire = 0;
	for (i = 0; i < N_NN; i++){	
		if (GE(t - T_last[i], Delta_abs)){
			v_dt = (-(v_nn[i] - E_leak) + R_nn * I_nn[i]) / Tau_m;	
			v_nn[i] += v_dt * T_step;
		}
		
		if (v_nn[i] > V_thresh){
			v_nn[i] = V_reset; T_last[i] = t;
			
			if (Num_id_fire == MAX_ID_FIRE) disp_err("err: NDa"); 
			Id_fire_v[Num_id_fire] = i; Num_id_fire++;

			send_syn_inp(i);
		}
	}
}

//presynaptic cell
void send_syn_inp(long i_pre){
	long i, k, n;
	
	for (i = 0; i < Syn[i_pre].num_post; i++){
		k = Syn[i_pre].id_post[i];
		n = Syn[i_pre].n_d[i];
		
		Arv[k][(M_arv + n)%MAX_ARV] += Syn[i_pre].w[i];	
	}
}

//random number
double std_gauss_ransuu(void){
	long i;
	double sum;
	sum = 0.e0;
	for (i = 0; i < 12; i++){
		sum += ransuu();
	}
	return (sum - 6.e0);
}

void my_fopen(FILE **fp, char fname[], long n_label){
	static long n_fop = 0;
	char text_wk1[TEXT_NUM], text_folder[TEXT_NUM];
	long i;	
	
	if (n_fop >= MAX_FILE_OPEN){ disp_err("err in my-fopen for opening too many files\n"); }
	if (n_label >= 2) disp_err("err my_fopen n-label\n");
	
	sprintf(text_wk1, "%s\\%s", OUT_FOLDER, fname);
	if (n_label == 1) strcat(text_wk1, Dlb_use);
	strcat(text_wk1, ".dat");
		
	if ( (*fp = fopen(text_wk1, "w") ) == NULL ){ 
		printf("err in fopcl for write mode open %s", text_wk1); 
		exit(1);
	}
	n_fop++;
}

void set_dlb_main_use(int argc, char *argv[]){
	if ( !strcmp(DLB_MAIN_BASE, "PRG") ){
		sprintf(Dlb_main, "(%s", argv[0]);
	}
	else{
		sprintf(Dlb_main, "(%s", DLB_MAIN_BASE);
	}
	
	strcat(Dlb_main, ")");
	strcpy(Dlb_use, Dlb_main);
}

float ransuu(void){
	if (RANSUU_METHOD == 2){
		return ran2_RCP(&I_ransuu);
	}
	else if (RANSUU_METHOD == 1){
		return ran1_RCP(&I_ransuu);
	}
	else if (RANSUU_METHOD == 0){
		return ran_fast_RCP(&I_ransuu);
	}
	else{
		printf("err in ran-suu\n"); 
		exit(1);
	}
}

void ransuu_init(long i_seed){
	if (i_seed >= 0){ printf("err i_seed >= 0\n"); exit(1); }
	I_ransuu = i_seed;
}

void disp_start_end_time(char md[], char c[]){
	FILE *fp;
	if ( (fp = fopen("time_now.dat", "a") ) == NULL){printf("err DPSET\n");exit(1); }
	
	c_time_now(Time_text);
	if (!strcmp(md, "s")){
		fprintf(fp, "start time of %s = %s\n", c, Time_text);
	}
	else{
		fprintf(fp, "end time of %s = %s\n", c, Time_text);
	}
	fclose(fp);
}

void disp_err(char c[]){
	printf("%s\n", c);
	exit(1);
}