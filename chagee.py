import streamlit as st
import pandas as pd
import random
import shap
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
from streamlit_carousel import carousel

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.linear_model import LogisticRegression  # logistic回归
from sklearn.neighbors import KNeighborsClassifier  # k近邻
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯
from sklearn.svm import SVC  # 支持向量机
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


random.seed(42)
st.set_page_config(page_title='SJTU', page_icon=None, initial_sidebar_state="expanded", layout='wide')
styles = {
    "container": {
        "margin": "0px !important",
        "padding": "0 !important",
        "align-items": "stretch",
        "background-color": "#fafafa"
    },
    "icon": {
        "color": "black",
        "font-size": "18px"
    }, 
    "nav-link": {
        "font-size": "18px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#fafa"
    },
    "nav-link-selected": {
        "background-color": "#ff4b4b",
        "font-size": "18px",
        "font-weight": "normal",
        "color": "black",
    },
}

def input_data():
    st.info("本应用为不记名自愿测试，您填写的所有的信息都将保密，希望您填写时不要有任何顾虑，为确保预测结果的真实准确，请按真实情况填写。")
    st.subheader(":blue[填写信息]")
    with st.expander("信息问卷", expanded=True):
        age, education, marriage, prep_will = 0, 0, 0, 0
        age = st.slider("1.您的年龄", 0, 100)
        edu_dict = {
            "小学及以下": 1, "初中": 2, "高中（含职高）": 3, "大学及以上": 4
        }
        education_text = st.selectbox("2.受教育程度",index=None, 
                                    options=edu_dict.keys(),placeholder="请选择")
        if education_text:
            education = edu_dict[education_text]
        marry_dict = {
            "单身": 1, "已婚": 2, "离异/丧偶": 3
        }
        marriage_text = st.selectbox("3.婚姻状态",index=None,
                                    options=marry_dict.keys(), placeholder="请选择")
        if marriage_text:
            marriage = marry_dict[marriage_text]
        st.text("4.是否使用过PrEP")
        prep_everuse = st.toggle(label="prep_everuse", label_visibility="collapsed")
        prep_everuse = int(prep_everuse)
        prep_dict = {
            "肯定不会": 1, "可能不会": 2, "不确定": 3, "可能会": 4, "肯定会": 5
        }
        prep_text = st.selectbox("5.PrEP使用意愿",index=None, 
                                    options=prep_dict.keys(),placeholder="请选择")
        if prep_text:
            prep_will = prep_dict[prep_text]
        st.text('6.是否发生过无保护肛交')
        uai = st.toggle(label="uai", label_visibility="collapsed")
        uai = int(uai)
        st.text('7.是否曾发生关系暴力')
        violence = st.toggle(label="violence", label_visibility='collapsed')
        violence = int(violence)

        social_support = st.slider("8.社会支持量表分数", 12, 84)
        sexualCompu = st.slider("9.性冲动分数", 10, 40)
        condom = st.slider("10.安全套使用技巧分数", 6, 30)
        condom_subj = st.slider("11.安全套使用主观规范分数", 0, 20)
        condom_effi = st.slider("12.安全套使用自我效能分数", 6, 30)
        st.text("13. 是否曾接受VCT检测")
        vct = st.toggle(label="vct", label_visibility='collapsed')
        vct = int(vct)
    
    data = {
        'Age': [age],
        'Education': [education],
        'Marriage': [marriage],
        'PrEPEverUse': [prep_everuse],
        'PrEPWill': [prep_will],
        'UAI': [uai],
        'Violence': [violence],
        'SocialSupport': [social_support],
        'SexualCompulsivity': [sexualCompu],
        'CondomSkill': [condom],
        'CondomSubjectiveNorm': [condom_subj],
        'CondomSelfEfficacy': [condom_effi],
        'VCTEverDone': [vct]
    }
    df = pd.DataFrame(data)
    return df
    

def load_csv_data(file_name):
    data = pd.read_csv(file_name)
    X = data.drop(['HIV'], axis=1)
    y = data['HIV']
    return X, y

def model_training(stacking_model, X_train, y_train, X_test):
    stacking_model.fit(X_train, y_train)
    y_prob = stacking_model.predict_proba(X_test)[:,1]
    return y_prob
def run_shap(stacking_model, X_train, X_test):
    explainer = shap.KernelExplainer(stacking_model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    fig = shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.success("分析完成！")
    st.pyplot(fig)

def run_model(X_test):
    file_name = '20240309_modelsmote3.csv'
    RANDOM_SEED = 42
    CV = 8
    X, y = load_csv_data(file_name)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=13)

    base_models = [
        LogisticRegression(penalty='l2', C=0.1, random_state=RANDOM_SEED),
        AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=RANDOM_SEED),
        KNeighborsClassifier(n_neighbors=4, p=2, weights="uniform"),
        MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(30, 20), random_state=RANDOM_SEED),
        LinearDiscriminantAnalysis()
    ]
    stacking_model = StackingCVClassifier(
        classifiers=base_models,
        use_probas=True,  # 使用概率作为元特征
        meta_classifier=LogisticRegression(penalty='l2', C=2, random_state=42),
        cv=8,
        random_state=42)
    with st.spinner("计算中......"):
        y_prob = model_training(stacking_model, X_train, y_train, X_test)
    st.success("计算完成！")
    output = f"您的HIV预测概率是: {y_prob}"
    st.markdown(output)
    with st.spinner("正在进行可解释性分析，预计等待1分钟，请稍候......"):
        run_shap(stacking_model, X_train, X_test)


# def predict(X_test):
#     st.markdown("您的信息输入如下")
#     st.write(X_test)
#     run_model(X_test)
    
    
def home():
    show()
    X_test = input_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        on_click = st.button("计算")
    if (on_click):
        run_model(X_test)
def question():
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#ac1736;'>社会支持量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/社会支持.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#ac1736;'>性冲动量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/性冲动.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#ac1736;'>安全套使用技巧量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/安全套使用技巧.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#ac1736;'>安全套使用主观规范量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/主观规范.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#ac1736;'>安全套使用自我效能量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/自我效能.png', use_column_width=True)
def about():
    html("""
    <style>
    iframe {
        margin-bottom: 0px;
    }
    html{
        margin-top: 0px;
        margin-bottom: 0px;
        border: 1px solid #de3f53;padding:0px 4px;    
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 400;
        line-height: 1.6;
        color: rgb(49, 51, 63);
        background-color: rgb(255, 255, 255);
        text-size-adjust: 100%;
        -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
        -webkit-font-smoothing: auto;
        }
        </style>
    
    <h2 style='color: #de3f53; margin-top:0px; border-bottom: solid 4px; '>更多心理咨询</h2>    
    
    <b>全国24小时心理热线</b>：<span style='color: #f2a93b; font-weight: bold;'>4001619995 <i class="fa-solid fa-calculator fa-fade"></i></span> <br/>
    <b>上海市24小时心理热线</b>：<span style='color: #f2a93b; font-weight: bold;'>962525 <i class="fa-solid fa-calculator fa-fade"></i></span> <br/>
    <b>上海市团市委电话心理咨询</b>：<span style='color: #f2a93b; font-weight: bold;'>021-12355 <i class="fa-solid fa-calculator fa-fade"></i></span>（工作日 9：00-21：00，双休日9：00-17：00）<br/>
    <b>上海市妇联心理关爱热线</b>：<span style='color: #f2a93b; font-weight: bold;'>021-12338 <i class="fa-solid fa-calculator fa-fade"></i></span>（周一至周五 9：00-17：30）<br/>
    """, height=190)
    html("""
    <style>
    iframe {
        margin-bottom: 0px;
    }
    html{
        margin-top: 0px;
        margin-bottom: 0px;
        border: 1px solid #de3f53;padding:0px 4px;    
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 400;
        line-height: 1.6;
        color: rgb(49, 51, 63);
        background-color: rgb(255, 255, 255);
        text-size-adjust: 100%;
        -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
        -webkit-font-smoothing: auto;
        }
        </style>
    
    <h2 style='color: #de3f53; margin-top:0px; border-bottom: solid 3px; '>更多HIV咨询、检测及就诊（上海地区）</h2>    
    
    <b>上海市自愿咨询检测机构名录</b>：<br/>
    <span style='color: blue; font-weight: bold;'>https://ncaids.chinacdc.cn/fazl/jcjg_10287/zyzxjcmz/201811/t20181113_197133.htm<i class="fa-solid fa-calculator fa-fade"></i></span> <br/>
    <b>上海艾滋病定点医院</b>： <br/>
    复旦大学附属华山医院-皮肤科<br/>
    上海交通大学医学院附属瑞金医院-皮肤科<br/>
    复旦大学附属中山医院-皮肤科<br/>
    上海交通大学医学院附属仁济医院-皮肤科<br/>
    上海交通大学医学院附属第九人民医院-皮肤科
    """, height=300)
    #st.info("如有需要，关注“上海市精神卫生中心”公众号预约就诊，地址：上海市徐汇区宛平南路600号")

    
    


menu = {
    'title': None,
    'items': { 
        '主页' : {
            'action': home,
            'item_icon': 'house',
        },
        '问卷量表' : {
            'action': question,
            'item_icon': 'bar-chart',
        },
        '健康服务' : {
            'action': about,
            'item_icon': 'people'
        },
    },
    'menu_icon': '', 
    'default_index': 0,
    'with_view_panel': 'sidebar',
    'orientation': 'vertical',
    'styles': styles
} 
def show_menu(menu):
    def _get_options(menu):
        options = list(menu['items'].keys())
        return options
    def _get_icons(menu):
        icons = [v['item_icon'] for _k, v in menu['items'].items()]
        return icons
    kwargs = {
        'menu_title': menu['title'],
        'options': _get_options(menu),
        'icons': _get_icons(menu),
        'menu_icon': menu['menu_icon'],
        'default_index': menu['default_index'],
        'orientation': menu['orientation'],
        'styles': menu['styles']
    }
    with_view_panel = menu['with_view_panel']
    if with_view_panel == 'sidebar':
        with st.sidebar:
            menu_selection = option_menu(**kwargs)
    elif with_view_panel == 'main':
        menu_selection = option_menu(**kwargs)
    else:
        raise ValueError(f"Unknown view panel value: {with_view_panel}. Must be 'sidebar' or 'main'.")
    if 'submenu' in menu['items'][menu_selection] and menu['items'][menu_selection]['submenu']:
        show_menu(menu['items'][menu_selection]['submenu'])
    if 'action' in menu['items'][menu_selection] and menu['items'][menu_selection]['action']:
        menu['items'][menu_selection]['action']()
def show():
    #st.image('banner2.png', use_column_width=True)
    st.markdown("""<h2 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#ac1736;'>跨性别女性HIV感染风险在线预测工具</span></h2>""", unsafe_allow_html=True)
    st.warning("通过回答以下问题，您可获得由模型在线计算出的HIV感染风险概率")
    st.image("img/roll_image1.png", width=1500)
    # st.markdown("""<h3 style='margin:0; text-align:centor; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#ac1736;'> </span></h3>""", unsafe_allow_html=True)
   

if __name__ == '__main__':
    
    st.sidebar.image('sjtu.png')
    show_menu(menu)
    st.sidebar.markdown('''
    <div style='padding:5px 5px 1px 5px; border-radius:8px; background-color: orange; color:white;'>
    <h4 style='color: white; padding:3px; font-weight:bold;'><i class="fa-solid fa-circle-chevron-right fa-fade"></i> 欢迎使用 </h4>
    <hr style='margin:0px 0px 5px 0px; padding:0; border: 2px solid white;'>
    - 基于沈阳跨性别女性群体调查数据

    <i class="fa-solid fa-language"></i> - Stacking机器学习算法构建 <br/>
    <i class="fa-solid fa-language"></i> - 上海交通大学医学院课题组开发 <br/>
    <i class="fa-solid fa-globe"></i> - 科学认识   平等包容
    </div>''', unsafe_allow_html=True)
    
    
