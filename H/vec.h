#define		VER				-2.e6
#define		NER				-20000
#define		VSQ				-30000.e0
#define		NSQ				-30000

long nbv(double vec[]);
long dip(double vec[], ...);
long dip_seq(double vec[], long max_v, double inp_v[]);
long nip(long vec[], ...);
long nip_seq(long vec[], long max_v, long inp_v[]);

//num by ver
long nbv(double vec[]){
    long i;
    for (i = 0; ;i++) if(vec[i] == VER) break;
    return i;
}

//example:
//if double a[10]; dip(a, 2.3e0, 3.e0, -4.e0, 6.7e0, -8.2e0, VER, 10);
//then a[0]=2.3e0, a[1]=3.e0, a[2]=-4.e0, a[3]=6.7e0, a[4]=-8.2e0, a[5]=VER
//if double a[10]; dip(a, 8.5e0, -1.5e0, 4.e0, VSQ, 10);
//then a[0]=8.5e0, a[1]=7.e0, a[2]=5.5e0, a[3]=4.e0; a[4]=VER
long dip(double vec[], ...){
	long i, max_v, num_v;
	long max_inp_v = 210;
	double *inp_v;
	inp_v = new double [max_inp_v];
	va_list ap;
	va_start(ap, vec);

	for (i = 0; ;i++){
		if (i == max_inp_v){ printf("err: d-ip too long\n"); exit(1); }
		inp_v[i] = va_arg(ap, double);
		if (inp_v[i] == VER){
			max_v = va_arg(ap, long);
			num_v = i;
			break;
		}
		if (inp_v[i] == VSQ){
			max_v = va_arg(ap, long);
			num_v = dip_seq(vec, max_v, inp_v);
			delete [] inp_v;
			return num_v;
		}
	}
	
	//error
	if (num_v+1 > max_v){
		printf("err:d-ip, vec[0]=%10.5le, vec[1]=%10.5le, vec[2]=%10.5le, num_v=%ld, max_v=%ld\n",
			inp_v[0], inp_v[1], inp_v[2], num_v, max_v);
		exit(1);
	}
	
	for (i = 0; i < num_v+1; i++) vec[i] = inp_v[i];
	
	va_end(ap);
	delete [] inp_v;
	return num_v;
}

//inp_v[0]: initial value, inp_v[1]: range, inp_v[2]: last value
long dip_seq(double vec[], long max_v, double inp_v[]){
	long i;
	double eps_y = 1.e-12;
	double x, x_i, delx, x_f;
	x_i = inp_v[0]; delx = inp_v[1]; x_f = inp_v[2];
	
	for (i = 0; ;i++){
		if (i == max_v-1){
			printf("err: dip-sq: x_i, del_x, x_f = %10.5le, %10.5le, %10.5le\n", x_i, delx, x_f);
			exit(1);
		}
		x = x_i + i * delx;
		if ( (delx > 0.e0) && (x > x_f + eps_y) ) break;
		if ( (delx <= 0.e0) && (x < x_f - eps_y) ) break;
		vec[i] = x;
	}
	vec[i] = VER;
	return (i);
}

//example:
//if long a[10]; nip(a, 2, 3, -4, 6, -8, NER, 10);
//then a[0]=2, a[1]=3, a[2]=-4, a[3]=6, a[4]=-8, a[5]=NER
//if long a[10]; nip(a, 9, -2, 3, NSQ, 10);
//then a[0]=9, a[1]=7, a[2]=5, a[3]=3; a[4]=NER
long nip(long vec[], ...){
	long i, max_v, num_v;
	long max_inp_v = 210;
	long *inp_v;
	inp_v = new long [max_inp_v];
	va_list ap;
	va_start(ap, vec);

	for (i = 0; ;i++){
		if (i == max_inp_v){ printf("err: d-ip too long\n"); exit(1); }
		inp_v[i] = va_arg(ap, long);
		if (inp_v[i] == NER){
			max_v = va_arg(ap, long);
			num_v = i;
			break;
		}
		if (inp_v[i] == NSQ){
			max_v = va_arg(ap, long);
			num_v = nip_seq(vec, max_v, inp_v);
			delete [] inp_v;
			return num_v;
		}
	}
	
	if (num_v+1 > max_v){
		printf("err:n-ip, vec[0]=%ld, vec[1]=%ld, vec[2]=%ld, num_v=%ld, max_v=%ld\n",
			inp_v[0], inp_v[1], inp_v[2], num_v, max_v);
		exit(1);
	}
	
	for (i = 0; i < num_v+1; i++) vec[i] = inp_v[i];
	
	va_end(ap);
	delete [] inp_v;
	return num_v;
}

//inp_v[0]: initial value, inp_v[1]: range, inp_v[2]: last value
long nip_seq(long vec[], long max_v, long inp_v[]){
	long i;
	long n, n_i, deln, n_f;
	n_i = inp_v[0]; deln = inp_v[1]; n_f = inp_v[2];
	
	for (i = 0; ;i++){
		if (i == max_v-1){
			printf("err: n-ip-sq: n_i, del_n, n_f = %ld, %ld, %ld\n", n_i, deln, n_f);
			exit(1);
		}
		n = n_i + i * deln;
		if ( (deln > 0) && (n > n_f) ) break;
		if ( (deln <= 0) && (n < n_f) ) break;
		vec[i] = n;
	}
	vec[i] = NER;
	return (i);
}
