#include <stdio.h>
#include <stdlib.h>
#define KPTSLEN 200
#define COUNTMAX 300

int CACULATES = 0;
void convolution(double data[], double filter[], int datalen, int filterlen,double dataf[])
{
	int i,j;
	int halflen = filterlen/2;
	for(i=0; i<datalen; i++)
	{
		double tmp = 0;
		for(j=-halflen; j<filterlen- halflen; j++)
		{
			if(i-j<datalen&& i-j>=0)
			{
				tmp =tmp + data[i-j]*filter[j+halflen];
			}
		}
		dataf[i] = tmp;
	}
}

void max(double data[], double cof[], int initial, int len)
{
	cof[0] = data[0+initial];
	cof[1] = 0;

	for(int i =1;i<len;i++)
	{
		if(data[i+initial]>cof[0])
		{
			cof[0] = data[i+initial];
			cof[1] = i;
		}
	}
}

void change_sampling_rate(double data[], int datalen, int samplingrate, double datasr[], int datalen_sr)
{
	double toratio = (double)samplingrate/100.0;
	double insertpos = 0;
	for(int i=0;i<datalen_sr;i++)
	{
		insertpos = toratio*i;
		int markd = insertpos;
		int marku = markd+1;
		double ratio = insertpos- double(markd);
		if(marku<datalen)
		{
			datasr[i] = (1-ratio)*data[markd] + ratio * data[marku];
		}
		else if(marku==datalen)
		{
			datasr[i] = data[datalen-1];
		}
	}
}

void ppg_fearure_extractor(double data[], double highpass[], double lowpass[], \
						    int datalen, int highlen, int lowlen, \
						    double **Fm, int samplingrate)
{
	int dl = datalen*100/samplingrate;
	double *datasr = (double *)malloc(dl*sizeof(double));
	if(samplingrate!=100){
		change_sampling_rate(data, datalen, samplingrate, datasr, dl);
		data = datasr;
		datalen = dl;
	}
	double T = 100;
	double maxslope = 150;
	int count = 0;
	int countfm = 0;
	int flag = -1;
	int mark = 0;
	double peak = 0;
	int tmpT = 0;
	double maxdif=0;
	double maxindex=0;
	double feature[COUNTMAX][10];
	double featuretmp[10] = {0};
	double datadif[datalen];
	double *cof = (double *)malloc(2*sizeof(double));
	double *dataf = (double *)malloc(datalen*sizeof(double));

	convolution(data, highpass, datalen, highlen, dataf);
	convolution(dataf, lowpass, datalen, lowlen, data);

	for(int i = 0; i<datalen; i++)
	{
		if(i>=4)
		{
			datadif[i] = (2.0*data[i] + data[i-1] -data[i-3] -2.0*data[i-4])/8.0;
			if(i>70)
			{
				max(datadif,cof,i-70,71);
				maxdif = cof[0];
				maxindex = cof[1];
				if(T*1.2<70)
				{
					tmpT = int(T*0.6);
					max(datadif,cof,i-35-tmpT,tmpT*2);
					maxdif = cof[0];
					maxindex = cof[1]-tmpT+35;
				}
				if(maxindex == 35 && maxdif> maxslope*0.7)
				{
					maxslope = maxslope*0.9+ maxdif*0.1;
					peak = i -35;
					if(featuretmp[0]!=0)
					{
						if(flag == 4)
						{
							feature[count][8] = featuretmp[0];
							feature[count][9] = featuretmp[1];
							//if((feature[count][8]-feature[count][0])>1.6*T&&(feature[count][8]-feature[count][0])<0.3*T)
							//{
							//	count--;
							//}
							//else
							//{
							if(count>=1)
							{
								double temp = feature[count][2]-feature[count-1][2];
								if(temp<=1.6*T&&temp>=0.3*T&&temp>=30&&temp<=160)
								{
									// fprint output to fea_times.txt
									double slopek = (feature[count-1][9]-feature[count][1])/(feature[count][8]-feature[count][0]);
									Fm[countfm][0] = (feature[count][2]-feature[count-1][2])/200;  // 200 is correct for FS=100 ..?
									Fm[countfm][1] = (feature[count-1][2]-feature[count-1][0])/200;  // 200 is correct for FS=100 ..?
									Fm[countfm][2] = (feature[count-1][8]-feature[count-1][2])/200;  // 200 is correct for FS=100 ..?
									Fm[countfm][3] = (feature[count-1][6]-feature[count-1][2])/200;  // 200 is correct for FS=100 ..?
									Fm[countfm][4] = (feature[count-1][4]-feature[count-1][2])/200;  // 200 is correct for FS=100 ..?
									Fm[countfm][5] = feature[count-1][3]-feature[count-1][1] - slopek*(feature[count-1][2]-feature[count-1][0]);
									Fm[countfm][6] = feature[count-1][5]-feature[count-1][1] - slopek*(feature[count-1][4]-feature[count-1][0]);
									Fm[countfm][7] = feature[count-1][7]-feature[count-1][1] - slopek*(feature[count-1][6]-feature[count-1][0]);
									countfm++;
								}
								T = T*0.9+(feature[count][2]-feature[count-1][2])*0.1;
							}
							//}
							count++;
						}
						if(countfm>=KPTSLEN-1||count>=COUNTMAX)
						{
							break;
						}
						feature[count][0] = featuretmp[0];
						feature[count][1] = featuretmp[1];
						flag = 1;
						mark = 1;
					}
				}
				if(datadif[i-35]>=0&&datadif[i-36]<0)
				{
					featuretmp[0] = i-35-2;
					featuretmp[1] = data[i-35-2];
				}
				if(flag==2&&double(i-35)>peak+T*0.1&&double(i-35)<peak+T*0.5)
				{
					max(datadif,cof,i-35-6,13);
					maxdif = cof[0];
					maxindex = cof[1];
					if(maxindex == 6)
					{
						if(maxdif<=0)
						{
							feature[count][4] = i-35-2;
							feature[count][5] = data[i-35-2];
							feature[count][6] = i-35-2;
							feature[count][7] = data[i-35-2];
							flag = 4;
						}
						else
						{
							feature[count][4] = featuretmp[0];
							feature[count][5] = featuretmp[1];
							flag = 3;
						}
					}
				}
				if(datadif[i-35]>=0&&datadif[i-34]<0)
				{
					if(flag==1)
					{
						feature[count][2] = i-35-2;
						feature[count][3] = data[i-35-2];
						flag = 2;
					}
					if(flag == 3)
					{
						feature[count][6] = i-35-2;
						feature[count][7] = data[i-35-2];
						flag = 4;
					}
				}
			}
		}
	}

	Fm[KPTSLEN-1][0] = countfm;
}


const int FS = 81;
int MAXDATALEN = FS * 60; // 60s length pulse segment


int main(int argc, char const *argv[])
{
	double **Fm2 =(double **)malloc(KPTSLEN*sizeof(double *));  // 特征点提取
	for (int i=0;i<KPTSLEN;i++) {
		Fm2[i] = (double *)malloc(8*sizeof(double));
	}

	// [1234] load txts:
	double buf[MAXDATALEN];  /*缓冲区*/
	FILE *fp;                /*输入文件指针*/
	FILE *fea_fp;            /*输出文件指针*/
	int count=0;             /*行字符个数*/
	double temp;
	int i = 0;

    // [1] 高通滤波器
	if((fp = fopen("/root/PulseWave_emotion_algorithm/bugu/PulseWave_emotion_algorithm/emo/ring/algorithm/ppg_bp_/utils/cpp-find-keypoints/highpass.txt","r")) == NULL) {
		perror("fail to read highpass");
		exit (1) ;
	}
	for(i=0;i<MAXDATALEN;i++) {
		if(fscanf(fp,"%lf\t",&temp)==-1) {
			break;
		}
		buf[i] = temp;
	}
	// printf("highpass len: %d\n",i);
	fclose(fp);

	int highlen =i;
	double highpass[i];
	for(int j = 0; j<i;j++) {
		highpass[j] = buf[j];
	}

    // [2] 低通滤波器
	if((fp = fopen("/root/PulseWave_emotion_algorithm/bugu/PulseWave_emotion_algorithm/emo/ring/algorithm/ppg_bp_/utils/cpp-find-keypoints/lowpass.txt","r")) == NULL) {
		perror("fail to read lowpass");
		exit (1) ;
	}
	for(i=0;i<MAXDATALEN;i++) {
		if(fscanf(fp,"%lf\t",&temp)==-1) {
			break;
		}
		buf[i] = temp;
	}
	// printf("lowpass len: %d\n",i);
	fclose(fp);

	int lowlen = i;
	double lowpass[i];
	for(int j = 0; j<i;j++) {
		lowpass[j] = buf[j];
	}

	// [3] pulse 序列
	if((fp = fopen("/root/PulseWave_emotion_algorithm/bugu/PulseWave_emotion_algorithm/emo/ring/algorithm/ppg_bp_/utils/cpp-find-keypoints/input.txt","r")) == NULL) {
		perror("fail to read input");
		exit (1) ;
	}
	for(i=0;i<MAXDATALEN;i++) {
		if(fscanf(fp,"%lf\t",&temp)==-1) {
			break;
		}
		buf[i] = temp;
	}
	// printf("input len: %d\n",i);
	fclose(fp);


	// [4] feature extractor
	int datalen = i;
	double data2[i];
	for(int j = 0; j<i;j++) {
		data2[j] = buf[j];
	}

	ppg_fearure_extractor(data2, highpass, lowpass, datalen, highlen, lowlen, \
							Fm2, FS);

	// [5] outputs
	fea_fp = fopen("/root/PulseWave_emotion_algorithm/bugu/PulseWave_emotion_algorithm/emo/ring/algorithm/ppg_bp_/utils/cpp-find-keypoints/fea.txt", "a");

	double len = Fm2[KPTSLEN-1][0];
	// printf("test2!!%lf\n", Fm2[KPTSLEN-1][0]);
	for(int j = 0; j<len;j++) {
		for(int k = 0 ; k<8;k++) {
			fprintf(fea_fp, "%lf\t",Fm2[j][k]);
		}
		fprintf(fea_fp, "\n");
	}
	fclose(fea_fp);

	return 0;
} 


