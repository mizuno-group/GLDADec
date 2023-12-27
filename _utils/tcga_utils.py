import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines.statistics import logrank_test
from lifelines import fitters as fit
from lifelines import KaplanMeierFitter as KMF

from sklearn.cluster import KMeans

from logging import getLogger
logger = getLogger(__name__)

class TCGA_Analysis():
    def __init__(self):
        self.deconv_res = None
    
    def set_data(self,exp,deconv_res,clinical):
        """
                    sample_1    sample_2    sample_3 
        TSPAN6	    36.7106	    23.4776	    90.4249	 
        TNMD	    14.2019	    1.1934	    0.0000	 
        DPM1	    64.9972	    82.4721	    110.8229	
        SCYL3	    5.6286	    12.3943	    31.8373	 
        C1orf112	2.5030	    10.7891	    15.4869	  
        """
        self.exp = exp
        self.deconv_res = deconv_res
        self.clinical = clinical

        logger.info('raw exp: {}'.format(self.exp.shape))
        logger.info('raw immune: {}'.format(self.deconv_res.shape))
        logger.info('raw clinical: {}'.format(self.clinical.shape))

    def preprocessing(self,hard_threshold=-1,lower_days=0,upper_days=3650):
        # sample name norm
        id2sample = pd.read_pickle('/workspace/github/GLDADec/data/TCGA/caseid2sample_dic.pkl')
        self.clinical.index = [id2sample.get(t) for t in self.clinical['case_id'].tolist()] # rename index

        exp_samples = self.exp.columns.tolist()
        target_clinical = self.clinical.loc[self.clinical.index.isin(exp_samples)]

        # process clinical information
        prognosis = target_clinical[["days_to_last_follow_up","vital_status"]]
        prognosis.columns = ["OS_Time","OS_Status"]
        def convert(x):
            try:
                return float(x)
            except:
                return np.nan
        fxn = lambda x : convert(x)
        prognosis["OS_Time"] = prognosis["OS_Time"].apply(fxn)

        prog_status = []
        status = prognosis["OS_Status"].tolist()
        for i in range(len(prognosis)):
            if status[i]=="Alive" or status[i]=="Not Reported":
                prog_status.append(0)
            elif status[i]=="Dead":
                prog_status.append(1)
            else:
                raise ValueError(" Inappropriate Status")
        prognosis["OS_Status"] = prog_status
        self.prognosis = prognosis

        # immune res profile
        if hard_threshold==-1:
            threshold = 1/len(self.deconv_res.T) # expected value
        else:
            threshold = hard_threshold
        fxn = lambda x : 1 if x > threshold else 0
        immune_binary = self.deconv_res.applymap(fxn)

        target_prognosis = prognosis.loc[prognosis.index.isin(self.deconv_res.index)]
        # FIXME: There may be a better processing method.
        target_prognosis['sample']=target_prognosis.index.tolist()
        target_prognosis = target_prognosis.drop_duplicates(keep='first')
        target_prognosis = target_prognosis.drop('sample',axis=1)

        concat_value_df = pd.concat([target_prognosis,self.deconv_res],axis=1)
        self.concat_value_df = concat_value_df[(0<concat_value_df['OS_Time']) & (concat_value_df['OS_Time']<upper_days)]

        concat_df = pd.concat([target_prognosis,immune_binary],axis=1)
        concat_df = concat_df.dropna()
        # duration selection
        self.concat_df = concat_df[(0<concat_df['OS_Time']) & (concat_df['OS_Time']<upper_days)]
    
    def calc(self,cell='Macrophage',do_plot=True):
        df1 = self.concat_df[self.concat_df[cell]==1]
        df0 = self.concat_df[self.concat_df[cell]==0]

        logger.info('positive samples: {}'.format(len(df1)))
        logger.info('negative samples: {}'.format(len(df0)))

        if min(len(df1),len(df0))==0:
            raise ValueError('Not stratified. Review threshold value.')

        # log-rank test
        results = logrank_test(df1["OS_Time"], df0["OS_Time"], df1["OS_Status"], df0["OS_Status"])
        p_log = float(results.summary["p"])

        # generalized wilcoxon test
        results = logrank_test(df1["OS_Time"], df0["OS_Time"], df1["OS_Status"], df0["OS_Status"], weightings = "wilcoxon")
        p_wil = float(results.summary["p"])

        self.res_log = pd.DataFrame({'log-rank':[p_log],'wilcoxon':[p_wil]})
        if do_plot:
            plot_curves(self.concat_df,target=[cell])
    
    def calc_top_bottom(self,cell='Macrophage',do_plot=True,title="Kaplan-Meier Plot",scale=4):
        concat_value_df = self.concat_value_df.dropna()
        cell_value = sorted(concat_value_df[cell].tolist())
        lower_threshold = cell_value[len(concat_value_df)//scale]
        upper_threshold = cell_value[-(len(concat_value_df)//scale)]

        def convert(x):
            if x > upper_threshold:
                return 1
            elif x < lower_threshold:
                return 0
            else:
                return np.nan
        fxn = lambda x : convert(x)

        concat_value_df[cell] = concat_value_df[cell].apply(fxn)
        concat_value_df = concat_value_df.dropna().astype(int)

        df1 = concat_value_df[concat_value_df[cell]==1]
        df0 = concat_value_df[concat_value_df[cell]==0]
        self.concat_final_value = concat_value_df

        logger.info('positive samples: {}'.format(len(df1)))
        logger.info('negative samples: {}'.format(len(df0)))

        if min(len(df1),len(df0))==0:
            raise ValueError('Not stratified. Review threshold value.')

        # log-rank test
        results = logrank_test(df1["OS_Time"], df0["OS_Time"], df1["OS_Status"], df0["OS_Status"])
        p_log = float(results.summary["p"])

        # generalized wilcoxon test
        results = logrank_test(df1["OS_Time"], df0["OS_Time"], df1["OS_Status"], df0["OS_Status"], weightings = "wilcoxon")
        p_wil = float(results.summary["p"])

        self.res_log = pd.DataFrame({'log-rank':[p_log],'wilcoxon':[p_wil]})
        if do_plot:
            plot_curves(concat_value_df,target=[cell],pvalue=p_log,title=title)


def plot_curves(df,target=[""],pvalue=None,title="Kaplan-Meier Plot"):
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)
    fig,ax = plt.subplots(dpi=300)
    lst = ["OS_Time","OS_Status"]+target
    df = df.loc[:,lst]
    cph = fit.coxph_fitter.CoxPHFitter()
    cph.fit(df, duration_col="OS_Time", event_col="OS_Status")
    cph.print_summary()
    if len(target)==1:
        plot_once(df,target[0])
    else:
        cph.plot_partial_effects_on_outcome(target, all_list(target))
    #plt.text(0.05,1.0,"$\it{P}$ = {}".format(str(pvalue)), transform=ax.transAxes, fontsize=10)
    plt.text(0.05,0.2,'$\it{P}$'+' = {}'.format(str(pvalue)), transform=ax.transAxes, fontsize=10) # for BRCA
    plt.title(title)
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.show()
    return

def plot_once(df,target,label_list=['Low (bottom 25%)','High (top 25%)']):
    ax = None
    for i, group in df.groupby(target):
        kmf = KMF()
        if len(label_list)==0:
            kmf.fit(group['OS_Time'], event_observed=group['OS_Status'],label = str(target) + ':' + str(i))
        else:
            kmf.fit(group['OS_Time'], event_observed=group['OS_Status'],label = label_list[i])
            
        if ax is None:
            ax = kmf.plot()
        else:
            ax = kmf.plot(ax=ax)


def plot_radar(data=[[0.3821, 0.6394, 0.8317, 0.7524],[0.4908, 0.7077, 0.8479, 0.7802]],labels=['Neutrophils', 'Monocytes', 'NK', 'Kupffer'],conditions=['BRCA','LUAD','LIHC'],title='Cell Proportions',show_name=True,figsize=(6,6),dpi=100):
    # preprocessing
    dft = pd.DataFrame(data,index=conditions,columns=labels).T
    pred = KMeans(n_clusters=len(conditions)).fit_predict(dft)
    dft['pred'] = pred
    dft = dft.sort_values('pred')
    labels = dft.index.tolist() # update
    dft = dft.T

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), dpi=dpi)

    # Helper function to plot each car on the radar chart.
    def add_to_radar(name, color):
        values = dft.loc[name].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.25)
    # Add each car to the chart.
    color_list = ['#429bf4','#f49b42','#38d583','#ec6e95','#8a4ebf']
    for ci in range(len(conditions)):
        add_to_radar(conditions[ci], color_list[ci])

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles)[0:len(labels)], labels)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    if show_name:
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')
    else:
        ax.set_xticklabels([i+1 for i in range(len(labels))])

    # Ensure radar range
    #ax.set_ylim(0, 0.9)
    ax.set_rlabel_position(180 / len(labels))
    ax.tick_params(colors='#222222')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(color='#AAAAAA')
    ax.spines['polar'].set_color('#222222')
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, y=1.1, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1),fontsize=15)
    plt.show()
