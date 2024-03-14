import streamlit as st
import pandas as pd
import random
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

        social_support = st.slider("8.社会支持量表分数", 0, 100)
        sexualCompu = st.slider("9.性冲动分数", 0, 50)
        condom = st.slider("10.安全套使用技巧分数", 0, 30)
        condom_subj = st.slider("11.安全套使用主观规范分数", 0, 20)
        condom_effi = st.slider("12.安全套使用自我效能分数", 0, 30)
        st.text("13. 是否曾接受VCT检测")
        vct = st.toggle(label="vct", label_visibility='collapsed')
        vct = int(vct)
    
    data = {
        'Age': [age],
        'Education': [education],
        'Marriage': [marriage],
        'PrEP_EverUse': [prep_everuse],
        'PrEP_Willingness': [prep_will],
        'UAI': [uai],
        'Violence': [violence],
        'SocialSupport_TotalScore': [social_support],
        'SexualCompulsivity_TotalScore': [sexualCompu],
        'Condom_Skill_TotalScore': [condom],
        'Condom_SubjectiveNorm_TotalScore': [condom_subj],
        'Condom_SelfEfficacy_TotalScore': [condom_effi],
        'VCTEverDone': [vct]
    }
    df = pd.DataFrame(data)
    return df
    

def load_csv_data(file_name):
    data = pd.read_csv(file_name)
    X = data.drop(['HIV'], axis=1)
    y = data['HIV']
    return X, y

def model_training(X_test):
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
    stacking_model.fit(X_train, y_train)
    y_prob = stacking_model.predict_proba(X_test)[:,1]
    return y_prob

def predict(X_test):
    st.markdown("您的信息输入如下")
    st.write(X_test)
    with st.spinner("计算中......"):
        y_prob = model_training(X_test)
    st.success("计算完成！")
    output = f"您的HIV预测概率是: {y_prob}"
    st.markdown(output)
def home():
    show()
    X_test = input_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        on_click = st.button("计算")
    if (on_click):
        predict(X_test)
def question():
    st.markdown("")
def about():
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#262261;'>社会支持量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/社会支持.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#262261;'>性冲动量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/性冲动.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#262261;'>安全套使用技巧量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/使用技巧.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#262261;'>安全套使用主观规范量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/主观规范.png', use_column_width=True)
    st.markdown("""<h3 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#262261;'>安全套使用自我效能量表</span></h3>""", unsafe_allow_html=True)
    st.image('question/自我效能.png', use_column_width=True)
    


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
    st.image('banner2.png', use_column_width=True)
    st.markdown("""<h2 style='margin:0; text-align:left; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#262261;'>跨性别女性HIV感染风险在线预测工具</span></h2>""", unsafe_allow_html=True)
    st.warning("通过回答以下问题，您可获得由模型在线计算出的HIV感染风险概率")
    st.markdown("""<h3 style='margin:0; text-align:centor; font-weight:bold;'> <i class="fa-solid fa-square-poll-vertical fa-beat"></i> <span style='color:#262261;'> </span></h3>""", unsafe_allow_html=True)
    test_items = [
    dict(title="",text="",interval=None,img="https://raw.githubusercontent.com/Suvirap/HIV_demo/main/img/roll_image3.png",),
    #dict(title="",text="",interval=2000,img="imgae3.png",),
    dict(title="",text="",interval=2000,img="https://raw.githubusercontent.com/Suvirap/HIV_demo/main/img/roll_image2.png",),
    dict(title="",text="",interval=None,img="https://raw.githubusercontent.com/Suvirap/HIV_demo/main/img/roll_image1.png",),
    ]
    
    carousel(items=test_items, width=1)

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
    <i class="fa-solid fa-globe"></i> - 科学认识 平等包容
    </div>''', unsafe_allow_html=True)
    
    
