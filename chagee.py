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
        condom_subj = st.slider("11.安全套使用主管规范分数", 0, 20)
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
    st.markdown('''知情同意书
您好！
我们将邀请您参加一项由上海交通大学公共卫生学院组织的研究调查，研究结果将有助于提供更好的健康服务，预防疾病。
本研究的目的是了解你的预防艾滋病相关的行为以及心理健康状况和影响因素。调查会以问卷调查的形式进行。如果您愿意参加本研究，我们的工作人员会同您一起完成问卷，大约花费您30-40分钟的时间。整个调查是以不记名的方式进行，您提供的所有数据都只会作为研究用途，只有研究人员能够接触到，并将在研究结束两年后销毁，因此您所有的资料都是绝对保密的。
本研究不会给您带来任何身体上的伤害。在调查过程中，您可以通过和工作人员沟通，了解到更多有关健康的知识。但调查中某些涉及心理和行为方面的问题可能会让您感到有点不舒服。
由于参加此次调查研究，将会占用您的时间和给您带来不便，您将会在完成调查后得到一定的数据采集的调查费和交通费。
您是否参加本次研究是完全自愿的，不会给您带来任何不良影响，即使在调查过程中，您也可以随时退出研究。如果您对本调查有任何疑问，请和研究人员联络。
同意参加调查研究
本人已阅读了知情同意书，并理解了同意书上的内容。现同意参加此项目，愿意接受该项目提供的服务并履行本人需要承担的义务。

参加者签名：                    日期：       年       月        日





亲爱的朋友，您好！
    这是一份健康相关的调查问卷，希望等得到您的支持。为了您个人信息的保密，本问卷为不记名自愿调查，希望您不要有任何顾虑，按真实情况填写，确保信息的真实有效，请认真阅读问题，并用笔在相应答案上打钩或者画圈，未注明的一律为单项选择题，只需要选择一个答案，谢谢合作！
A：基本情况
A1 您的出生年月：        年       月  （或您的年龄         周岁）
A2 您的文化程度：  
（1）小学及以下   （2）初中     （3）高中（含中专、职校） （4）大专及以上
(1) Primary school and below (2) Junior high school (3) Senior high school (including technical secondary school and vocational school) (4) Junior college and above
A3 您的婚姻状态：
（1）未婚         （2）已婚     （3）离婚                 （4）丧偶
(1) Unmarried (2) Married (3) Divorced (4) Widowed
A4 您每个月的平均收入（人民币）：
（1）3000元及以下 （2）3001-6000元 （3）6001-12000元 （4）12001元及以上
A5您在本地居住的时间：
（1）是本地居民    （2）不超过1年  （3） 1-5年       （4）超过5年
A6 您觉得自己的性取向属于：
（1）异性恋   （2）同性恋   （3）双性恋   （4）泛性恋   （5）不清楚________
A7 在过去六个月中，您是否接受过艾滋病预防的相关教育？（1）是      （2）否
A8 您是否愿意去做艾滋病自愿检测（VCT）？
（1）是             （2）否
A9 您是否曾做过艾滋病自愿检测（VCT）？
（1）是             （2）否
A10 您最近一次的艾滋病检测的结果：（1）阳性     （2）阴性     （3）不知道       
A11 过去30天您的吸烟状况：
（1）没有吸烟       （2）偶尔吸烟    （3）经常吸烟     （4）天天吸烟
B：跨性别情况
B1 您存在以下哪些情况？（可多选）
（1）穿女性衣服或化女妆  （2）女性化的面部整容手术 （3）使用雌激素
（4）隆胸             （5）切除男性全部或部分生殖器（6）重建女性生殖器官
（7）以上都没有（跳到B3）
B2 您发生以上行为(B1)的主要原因是：
（1）出于我的喜好  （2）是为了工作赚钱  （3）两者皆是  （4）其他          
B3 您对以上这些行为(B1)的认同度是：
（1）非常认同    （2）认同    （3）中立    （4）不认同    （5）非常不认同
B4 您周围的朋友对以上这些行为(B1)的认同度是：
（1）非常认同    （2）认同    （3）中立    （4）不认同    （5）非常不认同
B5 您的家人对以上这些行为(B1)的认同度是：
（1）非常认同    （2）认同    （3）中立    （4）不认同    （5）非常不认同
B6 您最尊敬的人对以上这些行为(B1)的认同度是：
（1）非常认同    （2）认同    （3）中立    （4）不认同    （5）非常不认同
C：艾滋病预防相关信息
C1：艾滋病预防相关知识
HIV-KQ-18	对	错	不知道
1. 咳嗽和打喷嚏不会传播艾滋病病毒	1	2	3
2. 与艾滋病病毒感染者共饮一杯水会被感染艾滋病病毒	1	2	3
3. 在射精前拔出性器官能避免伴侣感染艾滋病病毒	1	2	3
4. 与感染艾滋病病毒的男性发生肛门性交会感染艾滋病病毒	1	2	3
5. 性交后清洗生殖器官能避免感染艾滋病病毒	1	2	3
6. 艾滋病病毒阳性的孕妇必定会产下有艾滋病的婴儿	1	2	3
7. 感染了艾滋病病毒的人很快就会出现严重的感染症状	1	2	3
8. 现在有预防艾滋病病毒感染的疫苗	1	2	3
9. 与感染艾滋病病毒的伴侣深接吻、舌吻可能会感染艾滋病	1	2	3
10. 女性在月经期性交不会感染艾滋病病毒	1	2	3
11. 现在有女性避孕套可以降低女性感染艾滋病的机会	1	2	3
12. 正确和一贯使用安全套能降低艾滋病感染风险	1	2	3
13. 正在服用抗生素的人能够避免感染艾滋病病毒	1	2	3
14. 与多人发生性关系会增加感染艾滋病的机会	1	2	3
15. 性交后1周后进行检测能确认是否感染艾滋病病毒	1	2	3
16. 与艾滋病病毒感染者共浴或共享泳池会感染艾滋病病毒	1	2	3
17. 艾滋病病毒可以通过口腔性交传播	1	2	3
18. 在安全套上涂凡士林或婴儿油能降低感染艾滋病病毒风险	1	2	3
C2：艾滋病预防相关判断
艾滋病预防相关判断	非常不赞同	不赞同	无所谓	赞同	非常
赞同
1.对方看起来很健康，不可能有艾滋病	1	2	3	4	5
2.对方举止很绅士，不可能有艾滋病	1	2	3	4	5
3.我很了解的人不可能有艾滋病	1	2	3	4	5
4.对方说自己没有艾滋病，我会很相信	1	2	3	4	5

D：性行为和性工作相关问题
D1 风险感知
D11 您同意您是一个容易感染艾滋病的人吗？
（1）非常同意    （2）同意     （3）不知道  （4）不同意  （5）非常不同意
D12 您确定自己将来不会感染艾滋病吗？
（1）非常同意    （2）同意     （3）不知道  （4）不同意  （5）非常不同意
D13 您是否想到过某一天自己会感染艾滋病？
（1）从没有想到  （2）极少想到 （3）有时想到（4）经常想到（5）天天想到
D2 过去六个月的跨性别性行为
D21 在过去六个月中，您是否有固定的男性肛交行为的性伴（例如男朋友，不包括收费的商业性服务）？
（1）无（跳到D23）  （2）有
D22 在过去六个月中，您与固定男性性伴肛交时安全套的使用频率是：
（1）从来不用        （2）很少用      （3）大多时候用       （4）每次都用
D23 在过去六个月中，您是否有不固定的男性肛交行为的性伴（不含商业性服务）？
（1）无（跳到D25）  （2）有
D24 在过去六个月中，您与不固定男性性伴肛交时安全套的使用频率是：
（1）从来不用        （2）很少用      （3）大多时候用       （4）每次都用
D25 在过去六个月中，您是否有为男性顾客提供商业（收钱）的性服务？
（1）无（跳到D211）   （2）有
D26 在过去六个月中，您与男性客人肛交时安全套的使用频率是：
（1）从来不用   （2）很少用   （3）大多时候用   （4）每次都用（跳到D28）D27 与男性顾客肛交时没有使用安套的原因是：（可多选）
（1）顾客不想用 （2）自己不想用  （3）不用安全套可缩短肛交时间
（4）不用安全套可增加收入        （5）不用安全套可有更多客源 
（6）不用安全套可减少暴露自己身份的风险  （7）其他                
D28 您的男性客人的主要来源是：（可多选）
（1）酒吧     （2）桑拿浴室      （3）公厕     （4）社交活动    （5）公园 
（6）网络（微信、陌陌、遇见等）  （7）朋友介绍 
D29 您认为您周围的同行在与男顾客肛交时使用安全套的情况是：
（1）从来不用   （2）很少用     （3）大多时候用   （4）每次都用
D210 在过去一年中，您到过多少个中国其他城市去从事过男男的商业性服务？
（1）0          （2）1-3个      （3）3个以上
D211 您有无在性行为之前喝酒的习惯？
（1）从来不           （2）有时           （3）经常           （4）每次
D212 您有无在性行为之前服用毒品（摇头丸、冰毒或者大麻等）的习惯？ 
（1）从来不           （2）有时           （3）经常           （4）每次
D213 您的伴侣或顾客有无对您进行过身体袭击（如非自愿的打，拍，扇，推，踢等）？
（1）无       （2）有，是伴侣     （3）有，是顾客   （4）伴侣和顾客都有
D214 你的伴侣或顾客有无让您做了一些与您的性别认同不一致的事情（比如不让您使用您的化妆品）？
（1）无       （2）有，是伴侣     （3）有，是顾客   （4）伴侣和顾客都有
D215 您的伴侣或顾客有无对您进行过言语骚扰、诋毁或威胁？ 
（1）无       （2）有，是伴侣     （3）有，是顾客   （4）伴侣和顾客都有
D216 您的伴侣或顾客有无强迫过您发生性行为？ 
（1）无       （2）有，是伴侣     （3）有，是顾客   （4）伴侣和顾客都有
D217 在您18岁之前，有没有人威胁并强行和您发生性行为？
（1）无               （2）有           


D3 安全套的使用态度、主观规范及技巧
D31 安全套的使用态度
安全套使用态度	非常不赞同	不赞同	无所谓	赞同	非常
赞同
1. 我觉得用安全套肛交会令双方不舒服	1	2	3	4	5
2. 我认为安全套可以预防艾滋病的传播	1	2	3	4	5
3. 我认为安全套预防艾滋病并不可靠	1	2	3	4	5
4. 我觉得安全套使用起来太麻烦	1	2	3	4	5
5. 我觉得肛交时应该使用安全套	1	2	3	4	5
6. 我会尽量劝说对方使用安全套	1	2	3	4	5
D32 安全套的使用技巧
安全套使用技巧from CISQ-S	从不	极少	有时	多数	一直
1. 如果不使用安全套，我会拒绝发生肛交行为	1	2	3	4	5
2. 我会与对方一起讨论使用安全套的问题	1	2	3	4	5
3. 我会让对方明白如果没有安全套就不能肛交	1	2	3	4	5
4. 在肛交行为中我会要求全程使用安全套	1	2	3	4	5
5.我会告诉对方不用安全套会增加感染性病或艾滋病机会	1	2	3	4	5
6. 即使对方不愿使用安全套，我仍能说服对方使用安全套	1	2	3	4	5
D33 安全套使用的主观规范
安全套使用的主观规范 from TRA	非常不支持	不支持	不明确	支持	非常支持
1. 您的家里人对您在肛交时坚持使用安全套是什么态度？	1	2	3	4	5
2. 您的朋友对您在肛交时坚持使用安全套是什么态度？	1	2	3	4	5
3. 与您做相同工作的伙伴对您在肛交时坚持使用安全套是什么态度？	1	2	3	4	5
4. 对您来说重要的人（可以是家人朋友或其他任何人）对您在肛交时坚持使用安全套是什么态度？	1	2	3	4	5
D34安全套使用的自我效能
请判断你是否可以做到以下这些事情，圈出你认为最能反映你程度的对应数字，所有选项均无对错之分。	肯定无法做到	大概无法做到	也许可以也许不行	大概能做到	肯定能做到
1.我能够随身携带安全套以备不时之需。	1	2	3	4	5
2.每次发生肛交行为我都能使用新的安全套。	1	2	3	4	5
3.我能在肛交行为中会全程使用安全套。	1	2	3	4	5
4.我能在肛交行为中避免安全套的滑落。	1	2	3	4	5
5.我在饮酒的情况下也能与对方商讨使用安全套。	1	2	3	4	5
6.不论对方是谁，我都能与他商讨使用安全套。	1	2	3	4	5


E：心理状况
E1 Rosenberg Self Esteem Scale
请细读以下十个句子，并就你个人的情况，按合适的程度评分	很不
符合	不符合	符合	非常
符合
1. 我是一个有价值的人，至少与其他人在同一水平	1	2	3	4
2. 我感到我有许多好的品质	1	2	3	4
3. 总的来说，我认为自己是一个失败者	1	2	3	4
4. 我做事可以和大多人一样好	1	2	3	4
5. 我觉得自己没什么值得自豪的地方	1	2	3	4
6. 我对自己持一种肯定态度	1	2	3	4
7. 整体而言，我对自己觉得满意	1	2	3	4
8. 我要是能更看得起自己就好了	1	2	3	4
9. 我有时觉得自己确实很没用	1	2	3	4
10. 我认为自己一无是处	1	2	3	4
E2 UCLA Loneliness Scale
在以下每一句子中请圈出适当的数字来代表你的感受， 这些句子是没有正确或错误的答案。你只凭个人直接的感受来作答。请不要留空其中一句不作答。	从不	极少	偶尔	经常
1.我缺乏友伴	1	2	3	4
2.我没有一个可以求助的人	1	2	3	4
3.我是一个喜欢交际的人    30	1	2	3	4
4.我觉得被遗弃	1	2	3	4
5.我觉得被别人孤立	1	2	3	4
6.当有需要时，我都可以找到友伴  60	1	2	3	4
7.我因自己胆小而感到不快乐	1	2	3	4
8.虽然我四周都有人，但他们是不支持我	1	2	3	4
E3 The Patient Health Questionnaire (PHQ-9)
在过去2周里，以下这些问题困扰您的频率为？	完全
没有	有几天	超过一半天数	几乎每天
1. 做事提不起劲或没有兴致	1	2	3	4
2. 感觉心情很沮丧，很绝望	1	2	3	4
3. 入睡困难、睡不安稳或者睡眠过多	1	2	3	4
4. 感觉疲倦，没什么活力	1	2	3	4
5. 食欲不振或吃太多	1	2	3	4
6. 觉得自己很糟糕，活得很失败，让家人很失望	1	2	3	4
7. 很难集中精力做一件事情，比如读报或看电视	1	2	3	4
8. 感觉比平常更烦躁、坐立不安、动来动去	1	2	3	4
9. 有不如死掉或伤害自己的念头	1	2	3	4
E4 困顿感评估量表
仔细阅读每一个条目，并圈出你认为最能反映你程度的对应数字，请不要遗漏任何一句。	没有	很轻	中等	偏重	严重
1. 我觉得自己正陷入无法摆脱的困境中	0	1	2	3	4
2. 我强烈希望逃避我生活中的事物	0	1	2	3	4
3. 我处在一段无法摆脱的情感关系中	0	1	2	3	4
4. 我经常觉得我就想逃离	0	1	2	3	4
5. 我觉得无力改变一些事情	0	1	2	3	4
6. 我觉得被承担的义务困住，找不到出路	0	1	2	3	4
7. 我看不到现在自己有什么出路	0	1	2	3	4
8. 我想远离生活中其他对我很强势的人	0	1	2	3	4
9. 我强烈希望逃离我现在生活的环境和状态	0	1	2	3	4
10. 我觉得自己被其他人的行为和想法困住了	0	1	2	3	4
11. 我想摆脱（逃避）自己	0	1	2	3	4
12. 我无力改变自己	0	1	2	3	4
13. 我倾向于逃避面对自己的想法和感受 	0	1	2	3	4
14. 我觉得我被内心想法困住，找不到出路	0	1	2	3	4
15. 我想要彻底摆脱自己目前状态并重新开始	0	1	2	3	4
16. 我感觉自己困在一个的深洞里，无法逃离	0	1	2	3	4
E5 挫败感评估量表
以下是一些描述自我感受的语句。并圈出最能反映你在过去7天里感受的数字。请不要遗漏任何一句。	从不	极少	有时	经常	总是
1. 我觉得自己一事无成	0	1	2	3	4
2. 我觉得自己是一个成功的人	0	1	2	3	4
3. 我觉得自己被生活打败了	0	1	2	3	4
4. 总的来说，我觉得自己是个赢家	0	1	2	3	4
5. 我觉得在这个世界上没有我的立足之地	0	1	2	3	4
6. 我觉得在日常生活中我就是个出气筒	0	1	2	3	4
7. 我觉得很无力	0	1	2	3	4
8. 我觉得自己缺乏信心	0	1	2	3	4
9. 我觉得无论面对什么样的困难我都有能力应对	0	1	2	3	4
10. 我觉得自己处在了社会的底层	0	1	2	3	4
11. 我觉得自己被社会淘汰了，就像被踢出局了	0	1	2	3	4
12. 我觉得自己是生活中的失败者	0	1	2	3	4
13. 我觉得自暴自弃，已经放弃了自己	0	1	2	3	4
14. 我有沮丧和挫败的感觉	0	1	2	3	4
15. 我在争取生命中重要事情时的争斗是失败的	0	1	2	3	4
16. 我觉得自己毫无斗志	0	1	2	3	4
E6  INQ-15（本次研究的汉化版INQ-15问卷来自浙江大学心理学系陈树林教授团队，问卷的汉化和使用均经过原始英文版问卷设计者Van Orden教授授权）
下面一些问题是关于您最近几天（一周之内）的看法和感受，您不用考虑这些看法和感受的对与错，也不用考虑是否合适，只是按照您的直觉来回答，从1到7分，最高分表示非常符合，最低分表示一点也不符合。
最近这几天，我的感受是	一点也不符合- 部分符合- 非常符合
a1.如果我离开了，我身边的人会过的更好PB1	1	2	3	4	5	6	7
b2.如果没有我，我身边的人会更加幸福PB2	1	2	3	4	5	6	7
c3.我认为我是社会的负担PB3	1	2	3	4	5	6	7
d4.我认为我的死对身边的人来说是一种解脱PB4	1	2	3	4	5	6	7
e5.我认为我身边的人希望可以摆脱我PB5	1	2	3	4	5	6	7
f6.我认为我让身边的人的生活变得更加糟糕PB6	1	2	3	4	5	6	7
g7.其他人很关心我TB1	1	2	3	4	5	6	7
h8.我感觉有所归属TB2	1	2	3	4	5	6	7
i9.我很少与关心我的人有接触TB3	1	2	3	4	5	6	7
j10.我很幸运拥有很多关心我支持我的朋友TB4	1	2	3	4	5	6	7
k11.我感到与其他人失去了联结TB5	1	2	3	4	5	6	7
l12.我感觉我是个社交聚会的局外人TB6	1	2	3	4	5	6	7
m13.我感到在我需要帮助的时候有人能够帮助我TB7	1	2	3	4	5	6	7
n14.我与其他人关系密切TB8	1	2	3	4	5	6	7
o15.我每天至少有一次让我满意的人际交往TB9	1	2	3	4	5	6	7
E7  自杀能力评估
以下问题关于您的自杀想法和感受。请您根据自身情况，在最符合您目前实际情况的选项数字上画圈。从0到8分，最低分表示一点也不符合，最高分表示非常符合。
	一点也不符合- 部分符合- 非常符合
1.想象自己死了是件很恐怖的事10	0	1	2	3	4	5	6	7	8
2.我已经想象过自己最可能实施的自杀方式2	0	1	2	3	4	5	6	7	8
3.我现在承受痛苦的能力比以前大大提高3	0	1	2	3	4	5	6	7	8
4.尽管我有自杀想法，但不敢采取行动自杀40	0	1	2	3	4	5	6	7	8
5.我已经学会如何克服对疼痛的恐惧5	0	1	2	3	4	5	6	7	8
6.我尝试寻找更容易的自杀方法6	0	1	2	3	4	5	6	7	8
7.我在脑海中设想过死亡会是什么样子7	0	1	2	3	4	5	6	7	8
8.我经常一时兴起做某件事8	0	1	2	3	4	5	6	7	8
9. 我做事很冲动9	0	1	2	3	4	5	6	7	8
E8.1 您是否曾经考虑过自杀？                           （1） 是  （2） 否
E8.2 您是否曾经计划过自杀？                           （1） 是  （2） 否
E8.3 您是否曾经实施过（或者差点实施）自杀？           （1） 是  （2） 否
E8.4 过去一年内，您是否考虑过自杀？                   （1） 是  （2） 否
E8.5 过去一年内，您是否计划过自杀？                   （1） 是  （2） 否
E8.6 过去一年内，您是否实施过（或者差点实施）自杀？   （1） 是  （2） 否
E8.7 你的好友中有人曾经自杀或者故意伤害自己么？       （1） 是  （2） 否
E8.8 你的家人中有人曾经自杀或者故意伤害自己么？       （1） 是  （2） 否
F 社会支持：你与家人、朋友的关系怎么样？	很不同意
	不同意	较不同意	中立	较同意	同意	很同意
1.当你有需要的时候，总有一个好朋友在你身边	1	2	3	4	5	6	7
2.你有一个好朋友可同他/她分享开心或者不开心	1	2	3	4	5	6	7
3.你的家人真的十分愿意帮助你	1	2	3	4	5	6	7
4.你的家人可以给你心理上的支持	1	2	3	4	5	6	7
5.你有一个真的可以安慰你的朋友	1	2	3	4	5	6	7
6.你的朋友真的愿意帮助你	1	2	3	4	5	6	7
7.如果有什么事发生，你可以依靠你的朋友	1	2	3	4	5	6	7
8.你可以和家人诉说你自己的问题	1	2	3	4	5	6	7
9.你有一些朋友可以同他们分享开心或不开心	1	2	3	4	5	6	7
10.你生命中有个好朋友，他/她会关心你的感受	1	2	3	4	5	6	7
11.你的家人愿意和你一起做决定	1	2	3	4	5	6	7
12.你可以和朋友倾诉自己的问题	1	2	3	4	5	6	7
G  Sexual compulsivity
以下问题关于您在性相关方面的想法和感受。请您根据自身情况，在最符合您目前实际情况的选项数字上画圈。	非常不同意	不同意	同意	非常同意
1.我在性方面的欲望困扰了我的正常伴侣关系	1	2	3	4
2.我对性的想法和行为正在困扰着我的生活	1	2	3	4
3.我对性的渴望已经打乱了我的日常生活	1	2	3	4
4.我有时因为性行为而导致无法履行我的承诺和责任	1	2	3	4
5.我有时会一时性起而会失去控制	1	2	3	4
6.我发现自己工作时会想到性 	1	2	3	4
7.我觉得自己对性的想法和感受比实际能体验到的更强烈	1	2	3	4
8.我不得不努力去控制自己对性的想法和行为	1	2	3	4
9.我经常超过自己意愿而更多地想到性	1	2	3	4
10.我很难找到一个跟我同样对性非常渴望的性伴侣	1	2	3	4
H  本次四联快速检测的结果是（由调查员填写）
乙肝	丙肝	梅毒	艾滋病
（1）阳性	（1）阳性	（1）阳性	（1）阳性
（2）阴性	（2）阴性	（2）阴性	（2）阴性
                                    访问日期：          访问员:
    ''')
def about():
    st.markdown('about')

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
        '关于' : {
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
    <i class="fa-solid fa-globe"></i> - 促进对跨性别女性认知，构建平等社会
    </div>''', unsafe_allow_html=True)
    
    
