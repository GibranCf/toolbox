
#base
import pandas as pd
import numpy as np
from warnings import filterwarnings
# nlp
from nltk.tokenize import word_tokenize
# plot
filterwarnings("ignore")
from IPython.display import display
from plotly import express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
# sklearn
from sklearn.metrics import auc, roc_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score



def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}
def nonstopwords(text,stopwords,minleght= None):
    text_tokens = word_tokenize(text)
    if minleght is not None:
        text_tokens = [word for word in text_tokens if len(word) >= minleght]
        tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    return tokens_without_sw
# Class to create a model assessment, defined by a pipeline, X,y test and params to optimize
class modelAssess():
    def __init__(self,pipe,X_train,y_train):
        self.pipe = pipe
        self.X = X_train
        self.y = y_train
    def assess(self,param_name,param_range,cv,scoring=None):
        self.param_name = param_name
        self.param_range = param_range    
        self.cv = cv
        train_scores, test_scores = validation_curve(self.pipe,
                                                    self.X,
                                                    self.y,
                                                    param_name=param_name,
                                                    param_range=param_range,
                                                    cv=cv,scoring=scoring)
        self.train_scores = train_scores
        self.test_scores = test_scores
        train_scores_mean_cv = np.mean(train_scores,axis=1)
        test_scores_mean_cv = np.mean(test_scores,axis=1)
        scores=pd.DataFrame({param_name:list(param_range),'train_score':list(train_scores_mean_cv),'test_score':list(test_scores_mean_cv)})
        scores['diff']= abs(scores['train_score']-scores['test_score'])
        scores['optimal'] = scores['test_score'] - scores['diff'] 
        pscore = scores.melt(id_vars=[param_name])
        self.pscore = pscore
        optvals =pd.DataFrame(scores.apply(lambda x: x[param_name][x.optimal==x.optimal.max()][0],axis=1)).rename(columns={0:'optimal_'+param_name})
        self.optvals = optvals
        self.scores = scores
        self.param_name = param_name
        self.opt_value = self.scores[self.scores['optimal']==self.scores.max()['optimal']][self.param_name].iloc[0]
    def graph(self,x_scale='linear'):
        diff=self.pscore[(self.pscore['variable']=='diff')]
        train=self.pscore[(self.pscore['variable']=='train_score') ]
        test=self.pscore[(self.pscore['variable']=='test_score')]
        opt_value = self.opt_value
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Line(x=train[self.param_name], y=train['value'],
            name="train"),
            secondary_y=False,
        )

        fig.add_trace(

            go.Line(x=test[self.param_name], y=test['value'],
            name="test"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Line(x=diff[self.param_name], y=diff['value'],name="diff"),
            secondary_y=True,
        )

        fig.add_vrect(x0=opt_value, x1=opt_value, 
                    annotation_text="Optimal "+ self.param_name,
                    annotation=dict(font_size=14, font_family="Arial",font_color='purple'), opacity=0.25,
                    line_width=4,line_color='purple')

        # Add figure title
        fig.update_layout(
            title_text=f"Trade-off Variance-Bias <br><sup>{self.param_name}</sup>"
        )

        # Set x-axis title
        fig.update_xaxes(title_text=self.param_name,type=x_scale)
        # Set y-axes titles
        fig.update_yaxes(title_text="Score", secondary_y=False)
        fig.update_yaxes(title_text="<b style='color:green'>ΔScores</b> ",autorange="reversed", secondary_y=True)
        self.fig = fig 
        display(fig)      
def ConfusionMatrixPlotlyDisplay(matrix,classes_names,prob = True,colorscale='Viridis'):
    z = matrix
    x = classes_names
    y = classes_names
    if prob:
        z = z/z.sum()
        z_text = [[str(round(y,2)) for y in x] for x in z]
    else:
        z_text = [[str(y) for y in x] for x in z]
    zmin = 0
    zmax = z.sum()
    # change each element of z to type string for annotations
    if z.shape==(2,2):
        z_text[0][0] = 'TN = ' + z_text[0][0]
        z_text[1][1] = 'TP = '+ z_text[1][1]
        z_text[0][1] = 'FN = '+ z_text[0][1]
        z_text[1][0] = 'FP =' + z_text[1][0]
    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale=[[0, 'lightyellow'], [1, 'green']],zmin=zmin,zmax=zmax)

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>'
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    fig.update_xaxes(autorange='reversed')

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()
    return fig

def graphROC(y_test,y_probs):
    fpr, tpr, thresholds = roc_curve(y_test,y_probs[:,1])
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()
    return fig
def performanceMetrics(pipe,X_train,y_train,X_test,y_test,model):
    predictions = pd.DataFrame()
    pipe = pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    predictions=predictions.append(pd.DataFrame({'y_pred':list(y_pred),'model':[model]*len(y_pred)}))
    predictions['y']= y_test.reset_index().drop(columns='index')
    cf=confusion_matrix(predictions['y'],predictions['y_pred'],labels=[0,1])
    cfgraph = ConfusionMatrixPlotlyDisplay(cf, list(pipe.classes_),prob=True)
    y_probs = pipe.predict_proba(X_test)
    ROC = graphROC(y_test,y_probs)
    cfd=pd.DataFrame(cf)
    precision=cfd[1][1]/(cfd[1][1]+cfd[0][1]) 
    sensibility = cfd[1][1]/(cfd[1][1]+cfd[1][0])
    specifity =  cfd[0][0]/(cfd[0][0]+cfd[1][0])
    tfp = 1- specifity
    f1 = 2*((precision*sensibility)/(precision+sensibility))
    roc_auc = roc_auc_score(y_test,y_probs[:,1])
    metrics=pd.DataFrame({'metric':['precision','sensibility','specifity','tfp','f1','ROC AUC'],
                    'value':[precision,sensibility,specifity,tfp,f1,roc_auc]})
    display(metrics)
    return {'metrics':metrics,'confusion_matrix':cfd,'confusion_matrix_graph':cfgraph,
            'roc_curve':ROC,'roc_auc':roc_auc}

