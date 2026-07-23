# fqag064

Stylometric detection of AI-generated texts: evidence from
human and machine-written essays
Rongzhi Chen, Shizhao Xiong, Jingqi He, Gordon J Ross
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

Digital Scholarship in the Humanities, 2026, 00, 1–17
https://doi.org/10.1093/llc/fqag064
Full Paper
Stylometric detection of AI-generated
texts: evidence from human and
machine-written essays
Rongzhi Chen1, Shizhao Xiong1, Jingqi He1, and Gordon J. Ross1,*
1School of Mathematics, The University of Edinburgh, The King’s Buildings, Edinburgh EH9 3FD, UK
*Corresponding author. School of Mathematics, The University of Edinburgh, James Clerk Maxwell Building, Peter Guthrie Tait Road,
Edinburgh EH9 3FD, UK. E-mail: gordon.ross@ed.ac.uk
Abstract
The rise of large language models such as ChatGPT has intensified debates about authorship, originality,
and integrity in academic and creative writing. Distinguishing between human- and artificial intelligence
(AI)-generated texts is therefore not only a technical task but also a pressing concern for the digital humani-
ties, where style, creativity, and attribution remain central. In this study, we adapt stylometric techniques
to this new setting, introducing a dataset of paired human- and AI-authored essays across 110 subject
areas. We evaluate three established classifiers, analysing how essay length, training data size, and topical
variation influence performance, and whether human writing is better represented as a single category or
as distinct authorial voices. Our findings show that while AI-generated texts exhibit striking stylistic
uniformity, human writing is marked by variability and individuality. This contrast demonstrates both the
continuing effectiveness of stylometry for AI detection and its wider relevance for authorship, originality,
and voice in the age of generative AI.
Keywords: authorship attribution; function words; plagiarism detection; stylometry.
1. Introduction operationalize ‘style’ in measurable ways, linking compu-
tational analysis to long-standing scholarly questions
Stylometry is the quantitative analysis of linguistic style, about authorship and individuality.
long recognized in the digital humanities as a bridge be- The emergence of large language models (LLMs) such
tween computational techniques and interpretive ques- as ChatGPT has introduced a new domain where these
tions about authorship (Evert et al. 2017). Classical techniques are urgently needed. LLMs are increasingly
stylometry focuses on measurable textual features such used for automated content generation (Lu et al. 2023),
as word frequency distributions, vocabulary richness, code development (Biswas 2023), and academic writing
sentence length, and the use of function words. These (Imran and Almussharaf 2023). While these tools have
features serve as a distinctive stylistic fingerprint of an clear benefits, they also create challenges for originality,
author (Rudman 2006; Eder et al. 2016). attribution, and academic integrity. Detecting artificial
Although stylometry originated in literary and philo- intelligence (AI)-generated writing has therefore become
logical research, its role has broadened in digital humani- an important problem for educators, publishers, and
ties to include plagiarism detection, historical authorship researchers.
disputes (Burrows 2003; McCarthy and O’Sullivan 2021), Existing work on detecting AI-generated text has tradi-
and the study of stylistic development across periods and tionally relied on black-box machine learning classifiers
genres. What unites these applications is the ability to trained on large neural embeddings (Barlas and Stamatatos
© The Author(s) 2026. Published by Oxford University Press on behalf of EADH. This is an Open Access article distributed under the terms of the Creative Commons Attribution
License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is
properly cited.
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

2 Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
2020; Fabien et al. 2020), or more recently, zero-shot meth- conditions under which stylometry performs best, our
ods utilizing probability curvature (Mitchell et al. 2023; Wu work contributes both practical insights into AI text de-
et al. 2025). While these approaches can be effective, they tection and evidence of the continuing value of interpret-
offer limited interpretability and are not always accessible able, style-based methods within digital humanities
in a humanities research setting. Stylometry, by contrast, research.
provides transparent, lightweight methods grounded in a
long tradition of digital humanities research, making it a
2. Background and rationale
natural choice for our study.
Prior work comparing human and LLM writing has suc-
2.1 ChatGPT
cessfully demonstrated that linguistic differences are
detectable (Herbold et al. 2023). However, current litera-
ChatGPT is an AI model developed to generate human-
ture often prioritizes domain-specific case studies, such
like text, including essays, articles, and news reports.
as physics essays (Yeadon et al. 2023), or relies on com- The model is built on generative pre-trained transformer
plex, opaque deep-learning architectures (Berriche and (GPT) technology, which forms part of a broader class of
Larabi-Marie-Sainte 2024). While recent scholarship has LLMs specifically designed for natural language process-
begun to test the limits of stylometry on short texts ing (OpenAI 2022). Although our experiments employ
(Przystalski et al. 2026), there remains a lack of system- ChatGPT outputs as the AI-generated texts, the methods
atic research that isolates how these interpretable meth- are equally applicable to other LLMs, such as Claude
ods perform when multiple constraints, such as limited (Anthropic), Gemini (Google DeepMind), and Grok (xAI) if
training data, topic shifts, and authorship representa- the training data are adjusted accordingly.
tion, are varied simultaneously. We address this gap with
a paired-topic corpus and a systematic four-study design 2.2 Function words
that tests how these factors shape performance, yielding
actionable guidance and an interpretation of AI Function words are among the most established features
‘uniformity’ versus human variability. in stylometric analysis, due to their stability across an
Accordingly, this study pursues two objectives. First, author’s work and their unconscious use in writing
we evaluate the performance of stylometric methods in (Rudman 2006; Eder et al. 2016; Hossain et al. 2017).
distinguishing human from AI writing. Second, we exam- Unlike content words, function words contribute little
ine what these comparisons reveal about the shifting semantic meaning but are crucial for grammatical cohe-
notion of ‘authorship’ in digital culture. If machine- sion. Their extremely high frequency makes them espe-
generated texts display consistent stylistic patterns, cially useful for computational comparison: fewer than
while human writing is marked by diversity and idiosyn- 0.04 per cent of English vocabulary accounts for over
crasy, then authorship attribution becomes a way of ex- half of all words used in everyday discourse (Chung and
amining the contrast between uniformity in AI writing Pennebaker 2007). In this study, we use a curated set of
and variability in human writing. 70 function words, adapted from Mosteller and Wallace’s
This study addresses four such questions: (1) whether landmark analysis of The Federalist Papers (Mosteller
human writing is best represented as a single category and Wallace 1963), as listed in Table 1. In this study,
or as distinct authorial voices; (2) how essay length each essay is transformed into a 71-dimensional vector,
influences classifier performance; (3) how performance where the first 70 dimensions represent the proportion
shifts with limited training data; and (4) whether classi- of each function word in the text, with the final dimen-
fiers generalize across topics or rely on subject-specific sion representing the proportion of non-function words
cues. We evaluate three established stylometric meth- in the text.
ods—Burrows’ Delta, random forest (RF), and support We note that other feature representations—such as
vector machines (SVMs)—on a novel dataset of 4,346 character n-grams or neural embeddings derived from
paired human- and AI-authored essays. By clarifying the pre-trained language models like Fabien et al. (2020)and
Table 1. The 70 most frequently used function words in this analysis.
a, all, also, an, and, any, are, as, at, be, been, but, by, can, do, down, even, every, for, from, had, has, have, her, his,
if, in, into, is, it, its, may, more, must, my, no, not, now, of, on, one, only, or, our, shall, should, so, some, such, than,
that, the, their, then, there, things, this, to, up, upon, was, were, what, when, which, who, will, with, would, your
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00 3
Tyo et al. (2022)—have been shown to perform well in essay’s title to the GPT-4o mini language model. The fol-
authorship attribution tasks, particularly in PAN-style lowing prompt was used:
benchmarks. However, they are often difficult to inter-
pret and their connection to traditional stylistic analysis ‘Hi ChatGPT, I am going to give you a title for an es-
is less direct. Our aim here is not to pursue state-of-the- say, and I would like you to write a 1000-word essay
art accuracy, but to test whether function words, as a on this subject. Please do not include anything in
transparent and well-established stylometric feature set, your response except for the essay. Also, your essay
can reliably separate human and AI-generated texts. should not have section headings or a title. The ti-
Future work could extend this comparison to embed- tle is:’
dings or hybrid approaches, but our contribution is to es-
Using this instruction, a total of 2,173 essays were gener-
tablish a clear baseline within the stylometric tradition.
ated. While real-world use of AI often involves more var-
ied prompts and post-editing, we adopted a consistent
3. Data and visualizations
procedure to ensure comparability across topics and to
isolate stylistic differences between human and AI texts.
Our dataset comprises 4,346 essays, evenly split be-
tween 2,173 human-written and 2,173 ChatGPT-
generated entries. The human-written corpus was drawn 3.2 Comparative example of human
from the Aeon Essays Dataset (Acharya 2024), comprising and ChatGPT essays
long-form essays originally published by Aeon Media.
They cover 110 subject areas, ranging from the sciences To illustrate the differences in writing style between
and engineering to the arts and humanities, including human-authored and AI-generated texts, we present
topics such as ‘Architecture’, ‘Genetics’, and ‘Stories and the first 100 words from a human-written and an
Literature’. A detailed list of the analyzed topics is pro- AI-generated essay on the same topic: ‘Stories and
vided in Appendix A: List of Topics Analyzed. Literature’.
We treat each essay as originating from a unique au-
thor. This decision reflects the diversity of writing styles 3.2.1 Human-written essay (excerpt)
in the dataset and allows more precise modelling of vari-
In the surreal aftermath of my suicide attempt and amid
ation in human authorship. Aeon essays are particularly
the haze of my own processing, my best friend visited
suitable for our purposes as they cover a wide topical
me in the hospital with a (soft-bound and thus mental-
range, are authored by many distinct writers, and exhibit
patient-safe) copy of David Foster Wallace’s Infinite Jest
diverse stylistic characteristics. This diversity provides a
under his arm. It was the spring of 2021. A couple of
strong test case for distinguishing human writing from
months earlier, I had slipped in a tub, suffered a concus-
AI outputs.
sion, and triggered my first episode of major depression,
and those had been the most difficult months of my life.
3.1 ChatGPT-generated essays
Though a lifelong ‘striver’ and ‘high achiever’, nothing
I’ve ever done was harder than waging that war against
We selected GPT-4o mini for text generation to maxi-
myself while catatonic…
mize the ecological validity and temporal relevance of
our study. Conducted during the release window of the
3.2.2 ChatGPT-generated essay (excerpt) GPT-4o family, our data collection utilized what was
then the newest iteration of OpenAI’s model architec- Bereft and suicidal, I lay on my sofa, the weight of de-
ture. As a cost-effective and low-latency model, GPT-4o spair pressing down on me like an insurmountable force.
mini represents the class of LLMs most accessible to Each day felt like an echo of the last, a numbing repeti-
the general public and high-volume content generators. tion that drained the vibrancy from the world around
While larger models (such as GPT-4o or Claude 3.5 me. I found myself entrapped in a cycle of lethargy and
Sonnet) offer enhanced creative nuance, the ‘mini’ hopelessness, wandering through the dark corridors of
class accounts for a significant proportion of automated my mind. It was during these bleak hours that I stumbled
text currently circulating online due to its deployment in upon something surprising: a novel that would become
free-tier services. my lifeline. David Foster Wallace’s Infinite Jest emerged
For each human-written essay, we produced a corre- as a beacon in my darkness, its complexity and raw hon-
sponding ChatGPT-generated version by providing the esty tethering me to thoughts of life…
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

4 Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
3.3 Data visualizations potentially leading to more precise classifications by
reflecting the differences among various human authors.
To visualize relationships between texts in this feature
space, we employ multi-dimensional scaling (MDS;
Torgerson 1952), a dimensionality-reduction technique 4. Methodology
commonly used in stylometry (Ross 2020) that projects
high-dimensional observations into two dimensions 4.1 Introduction to
while preserving the original distance structure as
stylometry methods
closely as possible.
Figure 1 shows an MDS plot for our data, in which We now turn our attention to conducting a formal au-
each red point represents a ChatGPT-generated essay,
thorship attribution study. In this phase of our research,
while each blue point corresponds to a human-written
we will employ three standard methods of authorship at-
essay. This plot shows a clear separation between the
tribution that are widely recognized and commonly uti-
ChatGPT and human clusters. The ChatGPT cluster is
lized in the stylometry literature. These methods—
compact and tightly grouped, reflecting the model’s abil-
Burrows’ Delta (Delta), RF, and SVMs—each present
ity to produce uniform text across all generated essays.
unique advantages and are frequently applied to analyze
In contrast, the human cluster is more dispersed, indicat-
and classify authorship based on the linguistic and stylis-
ing greater variability in writing styles. This variability
tic features of texts. By comparing these methodologies,
likely reflects differences in individual authors’ linguistic
we aim to gain valuable insights into their respective
choices, levels of expertise, and personal interpretations
strengths and limitations, and identify the most suitable
of essay topics.
technique for our authorship attribution task.
A small number of red points overlap with the blue
cluster, suggesting that some ChatGPT-generated essays
closely resemble human writing. This raises the question 4.1.1 Burrows’ Delta
of whether to treat all human essays as a single repre-
Burrows’ Delta is a widely used method in classic stylom-
sentation of ‘human’ writing or to treat each essay as
etry (Hoover 2004; Argamon 2008; Evert et al. 2017), first
written by a distinct author. Given the observed over-
laps, the latter approach appears more appropriate. introduced by John F. Burrows as a simple yet effective
Treating all human essays as a single ‘blob’ assumes measure of stylistic difference based on word proportion
that human writing is relatively homogeneous. However, patterns (Burrows 2002).
the evident dispersion within the blue cluster suggests In our study, each essay is represented as a vector of
this assumption is inaccurate. The diversity of human function-word proportions. We then measure the stylis-
writing styles could compromise the accuracy of classifi- tic distance between texts using Euclidean distance, a
cation models. Treating each human essay as authored common variant of Delta that emphasizes larger devia-
by a separate individual acknowledges this variability, tions in word use. Classification is performed by
Figure 1. Multi-dimensional scaling plot visualising stylistic similarity among essays authored by humans and generated by
ChatGPT.
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00 5
assigning a text to the category with the closest aver- balances precision and recall. These metrics allow us to
age profile. evaluate not just whether classifiers are correct on aver-
The appeal of Delta lies in its interpretability: unlike age, but how they manage the trade-offs between false
more complex machine-learning models, it provides a positives (misclassifying human texts as AI) and false
transparent measure of stylistic similarity grounded in a negatives (failing to detect AI texts).
long-standing humanistic tradition. At the same time, its
simplicity makes it sensitive to data conditions, as we ex-
5. Results and discussion
plore in our experiments.
We conducted a series of experiments to assess how well
4.1.2 Support vector machines
the three classifiers—Burrows’ Delta (Delta), SVMs, and
RF—could distinguish human-written from ChatGPT- SVMs are a supervised machine learning method fre-
quently used in stylometric authorship attribution generated essays. All classifiers were implemented with
(Jockers and Witten 2010; Evert et al. 2017). The general standard parameter settings, tuned via cross-validation
principle is to construct a hyperplane that best separates where appropriate. Unless otherwise noted, classifiers
two classes of data points. In practice, many datasets were trained using a leave-one-out scheme so that each
are not linearly separable. To address this, SVMs employ essay was tested on models built from all remain-
kernel functions such as the radial basis function kernel, ing data.
which allows the model to capture subtle stylistic differ-
ences that cannot be represented with a straight line. 5.1 Study 1: aggregated versus
individual treatment of human-
4.1.3 Random Forest
written essays in authorship
(Breiman 2001) introduced RF as a powerful ensemble attribution
learning method, now widely used for both classification
and regression tasks. It operates by constructing multi- As can be seen in Fig. 1, human writing is inherently di-
ple decision trees during training and combining their verse, while ChatGPT-generated essays tend to be more
predictions to enhance accuracy and reduce over-fitting. uniform. We therefore compare two representations of
In stylometric analysis, RF has been applied to author- the human class: (1) an aggregated representation that
ship attribution by leveraging high-dimensional linguistic treats all human essays as if produced by a single ge-
features such as function word proportions, part-of- neric author, and (2) an individual-author representation
speech tags, and character n-grams (Stamatatos 2009; that treats each human essay as written by a distinct au-
Jockers and Witten 2010). Its ability to manage redun- thor. This comparison aims to determine whether ac-
dant or irrelevant features makes it well-suited to cap- counting for individual variability in human writing
turing the subtle stylistic cues that differentiate authors. improves classification performance. The corresponding
numerical values are reported in Appendix Tables 3and 4.
4.2 Introduction to
5.1.1 Human essays as a single
evaluation metrics
aggregated class
In machine learning and data science, evaluating a mod- We first assume that all human-written essays share a
el’s performance is crucial for understanding its effec- common authorship style, effectively collapsing them
tiveness. In our research, we use four standard metrics into a single class. We hence have a binary classification
(accuracy, precision, recall, and the F1 score) to assess task, with two classes: ‘Human’ and ‘AI’. Under this set-
classifier performance. Each captures a different aspect ting, the classifiers achieved moderate performance. As
of predictive ability, and together they provide a bal- shown in Fig. 2, RF performed particularly well, while
anced view of how well models distinguish between hu- Delta lagged behind. Evaluation metrics such as recall,
man- and ChatGPT-generated texts. precision, and F1 score aligned closely with accuracy.
Because our dataset contains an equal number of hu-
5.1.2 Human essays as separate authors
man and AI essays, accuracy serves as a useful overall
measure. However, accuracy alone can be misleading, so Next, we treat each human essay as if it were written by
we also report precision (the proportion of texts identi- a unique individual. Although the dataset contains ap-
fied as AI that truly are AI), recall (the proportion of AI proximately 1,500 actual authors, this method assumes
texts correctly identified), and the F1 score, which maximal stylistic variability by assigning a distinct author
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

6 Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
Figure 2. Results from Study 1: accuracy, precision, recall, and F1 scores for Burrows’ Delta, RF, and SVMs using combined and
separate datasets.
label to each human-written essay. It is important to representation with RF still offers a practical and
note that accuracy is still defined with respect to the bi- accurate alternative.
nary task of distinguishing human versus ChatGPT
essays. Consequently, if an essay written by one human 5.2 Study 2: impact of essay length
author is misclassified as belonging to a different human on classification performance
author, this still counts as a correct prediction.
Figure 2 shows this more granular approach yields Intuitively, we would expect essay length to have a direct
substantial improvements across all classifiers, clearly impact on classification accuracy. To assess this, we cre-
demonstrating that treating human-written essays as ated shortened versions of each essay by taking the first
authored by distinct individuals yields superior perfor- 200 words as a continuous block from the original text.
mance in distinguishing them from ChatGPT-generated This method preserves the natural flow and context that
texts. This approach better reflects the inherent stylistic occur in authentic writing. We then compared classifier
variability of human writing, enabling classifiers to cap- performance between these shortened essays and the
ture meaningful differences and reduce errors. full-length versions.
This contrast between the tight cluster of AI essays Figure 3shows that essay length strongly impacts per-
and the dispersed cluster of human essays highlights formance, with all classifiers experiencing declines
more than a technical classification problem. It reflects a across the four metrics when moving from full texts to
deeper distinction: human authorship is bound up with 200-word excerpts. Recall consistently showed the small-
individuality and variation, while AI writing projects a ho- est decline, indicating that even with shorter texts, clas-
mogenized style. Stylometry thus makes visible, in quan- sifiers could still identify ChatGPT essays with relative
titative form, a long-standing humanistic concern—the effectiveness, though at the expense of precision and
link between authorship and personal voice. overall balance. Precision experienced the largest de-
Consequently, we adopt the individual-author ap- crease, suggesting a greater likelihood of misclassifying
proach in all subsequent studies to maximize classifica- human-written essays as ChatGPT-generated in the
tion effectiveness. Nevertheless, in computationally shorter-text condition. This indicates that while the clas-
constrained scenarios, employing the aggregated ‘blob’ sifiers can capture the characteristics of ChatGPT-
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00 7
Figure 3. Results from Study 2: accuracy, precision, recall, and F1 scores for Burrows’ Delta, RF, and SVMs by essay length.
generated essays with shorter texts, they may also mis- also proved relatively resilient, though slightly less ro-
takenly identify features in human essays due to the lim- bust than SVMs in short-text scenarios.
ited stylistic data available. The underlying numerical
values for Delta, RF, and SVMs are reported in Appendix 5.3 Study 3: impact of training data
Tables 5–7. size on classifier performance
Delta and SVMs showed modest performance drops,
suggesting robustness to moderate reductions in text Training data availability is not a constraint, as ChatGPT
length. In contrast, RF exhibited the most pronounced texts can be generated in unlimited amounts and human
performance drop: while their recall showed almost no essays are readily accessible. The purpose is to show the
change across lengths, accuracy and precision fell to risks arising from small training sets, and to underline
about half of their original values and the F1 score why such conditions are not recommended.
dropped to about two-thirds of its original value in the Therefore, in this study, we examined how reducing the
200-word condition. This pattern indicates a strong bias number of training essays while keeping their original
towards predicting essays as ChatGPT-generated when length impacts the performance of these three classifiers.
text length is reduced, leading to many false positives. For each topic, we randomly reduced the number of avail-
As such, RF is not reliable as a stand-alone classifier for able training essays to 50 per cent, 20 per cent, and 10 per
short texts; if used for its high recall, it should be paired cent of the original count. For example, if the original
with a more accurate classifier to minimize errors. training set for a topic contained 100 essays, only 50, 20,
SVMs consistently outperformed the other classifiers or 10 essays were used in these reduced-data scenarios.
at both full and reduced lengths. This makes SVMs the Figure 4 shows how classifier performance varies with
most resilient option for datasets with varying or limited training set size. When trained on the full dataset, all classi-
text lengths, including condensed excerpts. fiers performed well, with RF achieving near-perfect results.
In summary, shorter essays generally reduced classi- The corresponding values for the 10, 20, and 50 per cent train-
fier performance, particularly in terms of precision. SVMs ing-data conditions are reported in Appendix Tables 8–10.
delivered the most stable and balanced results, while When the training data were reduced to 50 per cent,
RF’s high-recall but low-precision profile suggests its po- performance decreased slightly for most classifiers. RF
tential utility as a pre-classification filter. Burrows’ Delta remained the most robust, achieving an accuracy and F1
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

8 Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
Figure 4. Results from Study 3: accuracy, precision, recall, and F1 scores for Burrows’ Delta, RF, and SVMs using 10 per cent, 20 per
cent, 50 per cent, and 100 per cent of the training dataset (note: in Figure 4(a), the accuracy and recall lines overlap due to near-
identical values).
score of 99.82 per cent, showcasing stable performance Overall, these results show that RF delivers the stron-
even with fewer training examples. SVMs exhibited a gest performance across all training set sizes. Burrows’
more noticeable decline, with accuracy dropping to Delta also adapts relatively well, while SVMs are the
81.32 per cent and F1 score to 81.48 per cent, highlight- most sensitive to small training data. The improvement
ing its sensitivity to smaller datasets. Interestingly, of Burrows’ Delta at 50 per cent training size likely
Burrows’ Delta improved at the 50 per cent training size reflects a reduction in overfitting, as removing part of
(accuracy 99.54 per cent). the training data may eliminate noisy or atypical sam-
At 20 per cent training data, the gap between classi- ples and lead to cleaner frequency distributions.
fiers widened. RF continued to outperform the others, However, further reductions to 20 per cent and 10 per
maintaining accuracy and F1 score around 99.93 per cent deprive the method of sufficient information to cap-
cent. Visually, its lines appear almost flat, but the
ture stable stylistic patterns, causing performance to de-
reported values confirm a small decline from 99.82 per
cline. This suggests that Delta is sensitive not only to the
cent at 50 per cent to 99.31 per cent at 20 per cent.
amount of training data but also to its representative-
Burrows’ Delta experienced a moderate drop (F1
ness, performing best when the dataset strikes a balance
score=97.83 per cent). SVMs declined sharply to 79.08
between size and noise.
per cent accuracy and 78.87 per cent F1 score, and while
the downward slope looks modest in the plot, the nu-
5.4 Study 4: assessing the influence
merical difference highlights a clear loss of stability as
of topic relevance on the
data availability shrinks, indicating a growing vulnerabil-
ity as data availability is reduced. performance of classifiers
With only 10 per cent of the original training essays,
performance declined for all classifiers. RF still remained This experiment investigates whether high classification
the most resilient, with accuracy and F1 score around accuracy depends on training with essays from the same
98.4 per cent. Burrows’ Delta also showed reasonable ro- topic as the test data, or whether models can generalize
bustness (accuracy 90.58 per cent, F1 90.84 per cent). to unseen topics. In other words, we investigate whether
SVMs suffered the largest drop, with both accuracy and topic relevance is a necessary condition for achieving
F1 at around 75 per cent. strong performance. We therefore compare three
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00 9
scenarios: (1) training and testing on the same topic, 5.4.2 Scenario 2: using essays from a single
which evaluates how well classifiers exploit topic- unrelated topic
specific stylistic patterns; (2) training on a single unre-
In this scenario, classifiers were trained using essays
lated topic, which tests whether classifiers fail when the
from a single randomly chosen topic that differed from
training data bears no topical relation to the test essays;
the test topic.
and (3) training on all topics except the test topic, which As shown in Fig. 6, RF again outperformed the others,
assesses whether classifiers can generalize from broad achieving near-perfect metrics with accuracy, precision,
stylistic variation without direct exposure to the test recall, and F1 scores all around 99 per cent. SVMs dem-
topic. By contrasting these cases, we can evaluate the onstrated substantial improvement compared to
extent to which classifiers depend on topic-specific sty- Scenario 1, with all metrics exceeding 80 per cent, indi-
listic features versus topic-independent ones. cating better generalization across unrelated topics than
when restricted to topic-specific training data. In con-
5.4.1 Scenario 1: using essays from the trast, Burrows’ Delta achieved only moderate results (ac-
same topic curacy 65.74 per cent, recall 56.72 per cent, precision
56.27 per cent, and F1 51.74 per cent), underscoring its
Scenario 1 restricted both training and testing to the
reliance on topic-relevant training data. Due to the ran-
same topic, creating a stricter, topic-specific setting.
domness in topic selection, the performance of individ-
As shown in Fig. 5, RF demonstrated outstanding per-
ual topics in this scenario was not analyzed, as this
formance, with all four metrics consistently around 99
would provide limited additional insights.
per cent, indicating high reliability when topic-matched
training data were available. Burrows’ Delta also per-
formed well overall (93.5 per cent accuracy), but its
5.4.3 Scenario 3: using essays from all topics
results varied sharply by topic; for ‘Making’, where only
except the test topic
two essays per class were available, accuracy, precision,
recall, and F1 all fell to around 50 per cent, revealing sen- In this scenario, classifiers were trained on essays from
sitivity to extremely small samples. By contrast, SVMs all topics except the one being tested.
were the weakest on average (about 70 per cent across As shown in Fig. 7, both RF and SVMs achieved consis-
metrics). A notable exception was ‘Teaching and tently high performance, with accuracy, precision, recall,
Learning’, where SVMs achieved perfect scores, poten- and F1 scores all close to 99 per cent, demonstrating
tially due to this topic exhibiting distinctive and consis- strong generalization across diverse training data.
tent stylistic patterns. Burrows’ Delta also performed well, though slightly
Overall, under same-topic training, RF was robust, lower. RF exhibited near-perfect classification for most
Burrows’ Delta was competitive but data-sensitive, and topics, with only ‘Biography and Memoir’ and ‘History’
SVMs generally struggled except on highly distinc- recording slightly lower accuracies of 97 per cent and 99
tive topics. per cent, respectively.
Figure 5. Results from study 4: average accuracy, precision, recall, and F1 scores for Burrows’ Delta, RF, and SVMs in Scenario 1.
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

10 Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
Figure 6. Results from study 4: average accuracy, precision, recall, and F1 scores for Burrows’ Delta, RF, and SVMs in Scenario 2.
Figure 7. Results from study 4: average accuracy, precision, recall, and F1 scores for Burrows’ Delta, RF, and SVMs in Scenario 3.
5.4.4 Cross-scenario performance analysis robustness in identifying ChatGPT-generated essays, re-
gardless of the topic or size of the training data.
Analysing the performance of each classifier across
Burrows’ Delta performed poorly in Scenario 2 but
the three scenarios reveals several patterns, shown
achieved good results in Scenarios 1 and 3. This indicates in Fig. 8.
SVMs performed poorly in Scenario 1 but achieved that its performance is heavily influenced by the training
strong results in Scenarios 2 and 3. This outcome may be data size and relevance to the test topic. For these classi-
explained by the way SVMs construct their decision fiers, having essays from the same topic is critical when
boundaries. When the training essays all come from a data availability is limited. When same-topic training
single topic, topic-specific vocabulary, and phrasing data are unavailable, including more essays from diverse
dominate the feature space, masking the stylistic signals topics can partially mitigate this limitation.
that distinguish ChatGPT from human writing. As a result,
5.4.5 Conclusion
the support vectors chosen by the model reflect mainly
topic-related variation, which leads to overfitting and poor This experiment shows that topic relevance materially
generalization. In contrast, when trained on essays from di- shapes classifier performance. RF was consistently strongest
verse topics, the topic-specific noise is diluted and across all scenarios, making it the safest default and a reli-
ChatGPT’s consistent stylistic patterns become more promi- able option when no topic-matched training data are avail-
nent, enabling the model to classify more effectively. able. In contrast, SVMs performed poorly with the same-
In contrast, RF maintained consistently high perfor- topic training but improved markedly once training covered
mance across all three scenarios, demonstrating its diverse topics, indicating reliance on topic-independent
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00 11
Figure 8. Results from study 4: average accuracy, precision, recall, and F1 scores for Burrows’ Delta, RF, and SVMs across
different scenarios.
stylistic signals and vulnerability to topic-specific confounds. offs between robustness, sensitivity to data conditions,
Meanwhile, Burrows’ Delta was competitive with matched and interpretability. To support future applications, we
topics and broad coverage but fell to only moderate levels summarize the comparative strengths and weaknesses
when trained on a single unrelated topic, suggesting sensi- of each method below.
tivity to both topic relevance and sample size. Burrows’ Delta offered a balanced but sensitive pro-
From a practical perspective, when same-topic data file. It rarely achieved the highest accuracy observed
are scarce, RF should be regarded as the preferred with RF, but it remained competitive on full-length
method, with SVMs also suitable if the training corpus essays and provided the highest degree of interpretabil-
spans many topics. When limited same-topic data are ity. It also performed worse when essays by different
available, Burrows’ Delta can still be viable, provided authors were combined into a single class instead of
that sufficient in-topic examples are included. being treated as separate authors (Study 1). In addition,
The influence of the topic further underscores that au- its performance depended strongly on ‘topic relevance’.
thorship is not purely a matter of invariant linguistic fea- All metrics dropped sharply when the model was
tures. Human writers often express individuality through trained on unrelated topics (as shown in Study 4,
topic-specific vocabularies, while AI writing displays a Scenario 2). Delta is therefore well suited to digital hu-
cross-topic sameness. In this sense, the very success of manities research, where transparency is often priori-
stylometry in detecting AI rests on a paradox: the more tized over raw predictive performance, provided that
‘general’ the text appears, the less it resembles human the training data are topically aligned with the tar-
authorship, which is always situated and variable. get texts.
RF demonstrated the strongest overall performance,
5.5 Synthesis of classifier particularly in terms of robustness. It remained highly
performance across conditions accurate even when the training data were reduced to
10 per cent (Study 3) and generalized effectively across
Across the four studies, each classifier showed a distinct unrelated topics (Study 4, Scenario 2). It also performed
performance profile. These differences highlight trade- consistently well across different dataset structures in
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

12 Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
Table 2. Summary of classifier performance, strengths, and recommended use cases based on studies 1–4.
Classifier Key strengths Key weaknesses Best use case
Burrows’ Delta High interpretability (trans- Sensitive to digital humanities (DH) re-
parent feature usage); topic mismatch; search requiring
Competitive on Performance degrades on transparency;
matched topics. unrelated training data. Topically consis-
tent corpora.
RF Highest overall accuracy; Low precision on short General-purpose detection;
Robust to small train- texts (high false posi- Limited training data;
ing sets; tive rate); Unknown/mixed topics.
Generalizes well ‘Black box’ interpretability.
across topics.
SVMs Best performance on short Sensitive to small datasets; Short texts (e.g. stu-
texts (200 words); Prone to overfitting on sin- dent responses);
Strong generalization on gle-topic training data. Large, diverse train-
diverse corpora. ing corpora.
Study 1. However, it showed a clear limitation for short ChatGPT, our results demonstrate that standard RF clas-
texts (Study 2). In this condition, precision dropped sig- sifiers using simple function words can achieve compara-
nificantly, leading to a higher rate of false positives. This ble near-perfect accuracy, consistently over 98 per cent,
suggests that RF is the ideal default classifier for general- without the computational cost or opacity of neural net-
purpose detection, as long as the input texts are not works. This suggests that for many humanities applica-
too short. tions, classical machine learning remains a sufficient and
SVMs proved to be the most resilient method for short more resource-efficient tool.
texts (Study 2), outperforming other methods on 200- Second, our analysis of text length extends the con-
word excerpts. However, SVMs were highly sensitive to
clusions of Przystalski et al. (2026). Przystalski et al.
training conditions: they performed poorly when training
identified short text samples as a primary failure point
data were scarce (Study 3) or restricted to a single topic
for stylometric attribution. Our Study 2 confirms this
(Study 4, Scenario 1), likely due to overfitting on limited
general vulnerability but identifies a specific mitigation
feature sets. SVMs are therefore best suited for scenarios
strategy: while RF fails on short (200-word) texts due to
with abundant, diverse training data and shorter in-
high false positives, we found that SVMs remain remark-
put texts.
ably resilient in this condition. This distinction provides
Table 2provides a condensed overview of these find-
a practical correction to the literature: detection on ings to assist researchers in selecting the appropriate
short texts is not impossible, but requires shifting from
model for their specific constraints.
decision-tree methods to margin-based classifiers.
Finally, our confirmation of the ‘stylistic uniformity’ of
5.6 Contextualizing findings within
AI text expands upon Yeadon et al. (2023). Working
recent scholarship within the domain of physics essays, Yeadon et al. noted
that AI text lacks the variance of human writing. Our Our results confirm, distinctively refine, and in some
study generalizes this finding to the humanities (using
cases challenge recent findings in the rapidly evolving
field of AI detection. the Aeon corpus) and quantifies it through the perfor-
First, regarding classifier efficacy, our finding that RF mance leap observed in Study 1 (aggregated vs. individ-
consistently outperforms other methods aligns with the ual authorship). This reinforces the consensus that AI
broader trend in text classification but adds nuance to models function as ‘averaging machines’, producing a
the work of Berriche and Larabi-Marie-Sainte (2024). homogenized voice that is distinguishable not just by
While Berriche and Larabi-Marie-Sainte advocate for what it says, but by its lack of deviation from the statisti-
complex ensemble deep-learning models to detect cal mean.
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00 13
6. Limitations and future work allowed us to isolate clear stylistic signals and provide
transparent explanations of classifier behaviour, but fu-
This study was designed as a controlled evaluation, and ture work could expand the feature space to include
there are several directions in which the work can character n-grams, syntactic markers, or embeddings
be extended. (Tyo et al. 2022), enabling direct comparisons between
First, our AI texts were generated using a single, rela- interpretable baselines and higher-capacity representa-
tively simple prompt. This choice supports experimen-
tions. Relatedly, we evaluated three established classi-
tal control by isolating stylistic regularities attributable
fiers—Burrows’ Delta, RF, and SVMs—to balance
to the model rather than to heterogeneous user instruc-
classical stylometry with standard machine-learning
tions, but it does not fully reflect real-world AI use,
approaches; additional models and calibration strate-
where prompts vary widely and outputs are often
gies could further test the scope and limits of these
shaped through persona framing, iterative refinement,
conclusions.
and other forms of ‘prompt engineering’. Consequently,
Finally, although we focus on GPT-4o mini outputs in
our findings should be interpreted as a robust baseline
this study, replication across other generative models
under a controlled generation regime. Future work
and across multilingual contexts would provide a
should test whether these stylometric signals remain
broader assessment of generalizability and help disen-
stable under broader prompting conditions, including
tangle model-specific signatures from more stable prop-
alternative prompt families, genre-specific instructions,
erties of machine-generated text.
and adversarial prompting intended to reduce
Notwithstanding these important extensions, the pre-
detectability.
sent study establishes a critical baseline. By defining the
Second, our paired corpus—Aeon essays matched
scope of the current methods and controlling for topic,
with ChatGPT outputs by title—ensures topical compara-
we can more confidently interpret the fundamental di-
bility and supports clear interpretation of stylistic differ-
vergence observed in our results: the contrast between
ences. However, it represents one genre of relatively
the comparatively uniform stylistic regularities of ma-
polished long-form essays. Extending evaluation to addi-
chine generation and the inherent variability of hu-
tional domains (e.g. student coursework, journalistic
man expression.
writing, and shorter informal texts) would clarify how
well the observed patterns transfer across writing con-
texts and communicative settings. 7. Conclusion
Third, mixed-authorship (hybrid) writing is increas-
ingly common in scholarly and creative practice and is This study evaluated the effectiveness of stylometric
especially relevant for practical deployment. By ‘hybrid’ methods in distinguishing between human- and AI-
texts we refer to drafts in which human authors par- generated essays across four dimensions: whether hu-
tially rely on AI-generated passages and then revise, man texts are better treated as individual authors or a
paraphrase, or restructure them, as well as cases where single class; the influence of essay length; the impact of
AI rewrites human text. Although our experiments focus training data size; and the ability of classifiers to gener-
on the cleaner ‘pure human’ versus ‘pure AI’ setting to alize across topics. Using Burrows’ Delta, RF, and SVMs
establish internal validity, we do not evaluate hybrid on a balanced dataset of 4,346 human and ChatGPT
texts in the present study; however, future work could essays, we found that RF performed most strongly over-
test the same feature-extraction and classification pipe- all, with accuracy improving as essay length and train-
line on hybrid texts by varying human editing intensity ing size increased. Treating each human as a distinct
(light/medium/heavy) and the share of AI-generated author gave clearer separation than aggregating all hu-
content (25/50/75 per cent). We hypothesize that detec- man texts, and classifiers generalized reasonably well
tion performance may degrade as human revision across topics.
increases, and quantifying this trade-off would be cru- Beyond these methodological results, our findings
cial for real-world applications in digital humanities also highlight a more general pattern: AI-generated texts
and education. In addition, hybrid authorship moti- exhibited marked stylistic uniformity, while human writ-
vates more fine-grained methods than document-level ing was more variable and idiosyncratic. This contrast
classification: future work could adopt a rolling or win- helps to explain why stylometric methods remain effec-
dowed analysis, applying classifiers to sequential seg- tive for AI detection, but it also has broader implications.
ments to detect internal shifts in authorial voice. For digital humanities, variability of style has long been
Fourth, we focused on function words, a classic and central to how authorship is identified, contested, and
interpretable feature set in stylometry. This choice valued. In this sense, stylometry remains relevant not
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026

14                                                                                                                                                                           Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
Funding
only as a technical tool for classification, but also as a
way of reflecting on the distinctiveness of human writing
| in the age of generative AI. |     |     |     |     |     | None declared. |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
Future work should test the robustness of these find-
ings on other genres and corpora, including mixed-  References
authorship texts, student writing, and AI-generated out-
puts from models beyond ChatGPT. Expanding to multi- Acharya, M. (2024) Aeon Essays Dataset, https://www.
Downloaded from https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041 by guest on 23 July 2026
lingual contexts and incorporating additional feature  kaggle.com/dsv/7371248, accessed 13 Oct. 2024.
sets would further clarify the scope and limits of stylo- Argamon,  S.  (2008)  ‘Interpreting  Burrows’s  Delta:
metric approaches. Geometric  and  Probabilistic  Foundations’,  Literary
and Linguistic Computing, 23: 131–47.
7.1 Implications for authorship and  Barlas, G., and Stamatatos, E. (2020) ‘Cross-Domain
|     |     |     |     |     |     | Authorship  | Attribution  |     | Using  | Pre-Trained  | Language  |     |
| --- | --- | --- | --- | --- | --- | ----------- | ------------ | --- | ------ | ------------ | --------- | --- |
the human-AI nexus
Models’, in Maglogiannis, I., Iliadis, L., Pimenidis, E.
Beyond classification accuracy, our findings speak to  (eds)  Artificial  Intelligence  Applications  and
Innovations, pp. 255–266. Cham: Springer.
current questions about authorship and cultural value
in the humanities. We show that AI-generated essays  Berriche,  L.,  and  Larabi-Marie-Sainte,  S.  (2024)
exhibit comparatively stable stylistic patterns while hu- ‘Unveiling Chat-GPT Text Using Writing Style’, Heliyon,
10: e32976.
man essays display substantial dispersion and individ-
ual  variation,  making  ‘voice’  legible  in  quantitative  Biswas,  S.  (2023)  ‘Role  of  ChatGPT  in  Computer
form. This matters socially because LLMs are already  Programming’,  Mesopotamian  Journal  of  Computer
Science, 2023: 9–15.
used in educational and professional writing contexts,
creating practical tensions around originality, attribu- Breiman, L. (2001) ‘Random Forests’, Machine Learning,
| tion, and integrity. |     |     |     |     |     | 45: 5–32. |     |     |     |     |     |     |
| -------------------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- |
Interpretable stylometric features can therefore serve  Burrows,  J.  (2002)  ‘Delta’:  A  Measure  of  Stylistic
Difference and a Guide to Likely Authorship’, Literary
as a lightweight screening tool for educators, editors,
and researchers, especially when topic-matched training  and Linguistic Computing, 17: 267–87.
data are scarce, while also providing a DH-relevant per- Burrows, J. (2003) ‘Questions of Authorship: Attribution
and beyond a Lecture Delivered on the Occasion of
spective on how uniformity and variability function as
markers of authorship. the Roberto Busa Award ACH-ALLC 2001, New York’,
At the same time, such tools should be treated as prob- Computers and the Humanities, 37: 5–32.
|     |     |     |     |     |     | Chung,  C.,  | and  | Pennebaker,  |     | J.  | (2007)  | ‘The  |
| --- | --- | --- | --- | --- | --- | ------------ | ---- | ------------ | --- | --- | ------- | ----- |
abilistic indicators rather than definitive proof. In practice,
they should be used in a human-in-the-loop process: out- Psychological  Functions  of  Function  Words’,  in  K
puts support transparency and auditability, but final judg- Fiedler (ed.) Social Communication, pp. 343–359. New
ments  require  human  review,  particularly  as  mixed-  York: Psychology Press.
authorship writing becomes increasingly common. Eder,  M.,  Rybicki,  J.,  and  Kestemont,  M.  (2016)
‘Stylometry with R: A Package for Computational Text
Analysis’, The R Journal, 8: 107–21.
Author contributions
Evert, S. et al. (2017) ‘Understanding and Explaining
|                 |                      |     |     |                    |     | Delta  Measures  |     | for  Authorship  |     | Attribution’,  |     | Digital  |
| --------------- | -------------------- | --- | --- | ------------------ | --- | ---------------- | --- | ---------------- | --- | -------------- | --- | -------- |
| Shizhao  Xiong  | (Conceptualization,  |     |     | Formal  analysis,  |     |                  |     |                  |     |                |     |          |
Scholarship in the Humanities, : ii4–ii16.
Methodology, Project administration, Software, Writing—
Fabien, M. et al. (2020) ‘BertAA: BERT Fine-Tuning for
original draft, Writing—review & editing), Rongzhi Chen
Authorship Attribution’, in Bhattacharyya, P., Sharma,
| (Conceptualization,  |     | Data  | curation,  | Formal  analysis,  |     |     |     |     |     |     |     |     |
| -------------------- | --- | ----- | ---------- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
D. M., and Sangal, R. (eds), Proceedings of the 17th
| Investigation,  | Methodology,  |     | Resources,  | Visualization,  |     |                |     |             |     |          |           |     |
| --------------- | ------------- | --- | ----------- | --------------- | --- | -------------- | --- | ----------- | --- | -------- | --------- | --- |
|                 |               |     |             |                 |     | International  |     | Conference  | on  | Natural  | Language  |     |
Writing—original draft, Writing—review & editing), Jingqi
|                          |              |                 |             |                 |     | Processing   | (ICON),  | pp.              | 127–137.  | Patna,  | India:     | NLP  |
| ------------------------ | ------------ | --------------- | ----------- | --------------- | --- | ------------ | -------- | ---------------- | --------- | ------- | ---------- | ---- |
| He  (Conceptualization,  |              | Visualization,  |             | Writing—review  | &   |              |          |                  |           |         |            |      |
|                          |              |                 |             |                 |     | Association  | of       | India  (NLPAI),  |           | Indian  | Institute  | of   |
| editing),                | and  Gordon  | Ross            | (Software,  | Supervision,    |     |              |          |                  |           |         |            |      |
Technology Patna: .
Writing—review & editing)
Herbold, S. et al. (2023) ‘A Large-Scale Comparison of
|                       |     |     |     |     |     | Human-Written                  |     | versus  | ChatGPT-Generated  |     | Essays’,  |     |
| --------------------- | --- | --- | --- | --- | --- | ------------------------------ | --- | ------- | ------------------ | --- | --------- | --- |
| Conflicts of interest |     |     |     |     |     | Scientific Reports, 13: 18617. |     |         |                    |     |           |     |
Hoover, D. L. (2004) ‘Testing Burrows’s Delta’, Literary
| None declared. |     |     |     |     |     | and Linguistic Computing, 19: 453–75. |     |     |     |     |     |     |
| -------------- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- |

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00                                                                                                                                                                           15
Hossain, M. T. et al. (2017) ‘A stylometric analysis on  Yeadon, W. et al. (2023) ‘The Death of the Short-Form
Bengali  literature  for  authorship  attribution’,  in  Physics Essay in the Coming AI Revolution’, Physics
Proceedings of the 2017 20th International Conference  Education, 58: 035027.
of Computer and Information Technology (ICCIT), 1–5.
Imran, M., and Almussharaf, N. M. (2023) ‘Analyzing the
Appendix A: list of topics
| Role  of  | ChatGPT  | as  a  | Writing  | Assistant  | at  Higher  |     |     |     |     |     |     |     |
| --------- | -------- | ------ | -------- | ---------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
Education Level: A Systematic Review of the Literature’,  analyzed
Downloaded from https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041 by guest on 23 July 2026
Contemporary Educational Technology, 15: ep464.
Jockers, M. L., and Witten, D. M. (2010) ‘A Comparative  Addiction,  Ageing  and  death,  Animals  and  humans,
Study  of  Machine  Learning  Methods  for  Authorship  Anthropology,  Archaeology,  Architecture,  Art,  Astronomy,
Attribution’, Literary and Linguistic Computing, 25: 215–23.
Automation and robotics, Beauty and aesthetics, Bioethics,
Lu, G., Larcher, S. B., and Tran, T. (2023) Hybrid long  Biography and memoir, Biology, Biotechnology, Chemistry,
document summarization using C2F-FAR and ChatGPT:
|     |     |     |     |     |     | Childhood  | and  adolescence,  |     | Cities,  | Cognition  |     | and  intelli- |
| --- | --- | --- | --- | --- | --- | ---------- | ------------------ | --- | -------- | ---------- | --- | ------------- |
a practical study, arXiv: 2306.01169 [cs.CL], preprint:
gence, Comparative Philosophy, Complexity, Computing and
not peer reviewed. artificial  intelligence,  Consciousness  and  altered  states,
McCarthy, R., and O’Sullivan, J. (2021) ‘Who Wrote
|            |             |     |          |              |          | Cosmology,  | Cosmopolitanism,  |     | Dance  | and  | theatre,  | Death,  |
| ---------- | ----------- | --- | -------- | ------------ | -------- | ----------- | ----------------- | --- | ------ | ---- | --------- | ------- |
| Wuthering  | Heights?’,  |     | Digital  | Scholarship  | in  the  |             |                   |     |        |      |           |         |
Deep time, Demography and migration, Design and fashion,
Humanities, 36: 383–91.
Earth science and climate, Ecology and environmental scien-
Mitchell, E. et al. (2023) ‘DetectGPT: Zero-Shot Machine-
ces, Economic history, Economics, Education, Engineering,
Generated Text Detection using Probability Curvature’,  Environmental history, Ethics, Evolution, Fairness and quality,
| in  Krause,  | A.,  | Brunskill,  | E.,  Cho,  | K.,  Engelhardt,  | B.,  |     |     |     |     |     |     |     |
| ------------ | ---- | ----------- | ---------- | ----------------- | ---- | --- | --- | --- | --- | --- | --- | --- |
Family life, Film and visual culture, Food and drink, Future of
Sabato, S., and Scarlett, J. (eds), Proceedings of the
technology, Genetics, Global history, History, History of ideas,
40th International Conference on Machine Learning, vol.  History of science, History of technology, Home, Human evo-
202, pp. 24950–62. Honolulu, HI: PMLR.
|     |     |     |     |     |     | lution,  Human  | reproduction,  |     | Human  |     | rights  | and  justice,  |
| --- | --- | --- | --- | --- | --- | --------------- | -------------- | --- | ------ | --- | ------- | -------------- |
Mosteller, F., and Wallace, D. L. (1963) ‘Inference in an
|             |            |     |          |          |           | Illness  and  | disease,  | Information  |     | and  | communication,  |     |
| ----------- | ---------- | --- | -------- | -------- | --------- | ------------- | --------- | ------------ | --- | ---- | --------------- | --- |
| Authorship  | Problem’,  |     | Journal  | of  the  | American  |               |           |              |     |      |                 |     |
Knowledge, Language and linguistics, Life stages, Logic and
Statistical Association, 58: 275–309.
|     |     |     |     |     |     | probability,  | Love  | and  | friendship,  | Making,  | Mathematics,  |     |
| --- | --- | --- | --- | --- | --- | ------------- | ----- | ---- | ------------ | -------- | ------------- | --- |
OpenAI (2022) Introducing ChatGPT https://openai.com/
|     |     |     |     |     |     | Meaning  | and  the  | good  | life,  | Medicine,  | Mental  | health,  |
| --- | --- | --- | --- | --- | --- | -------- | --------- | ----- | ------ | ---------- | ------- | -------- |
blog/chatgpt, accessed 10 November 2024.
|               |     |                  |              |     |             | Metaphysics,  | Mood    | and  | emotion,    | Music,  | Nations          | and  |
| ------------- | --- | ---------------- | ------------ | --- | ----------- | ------------- | ------- | ---- | ----------- | ------- | ---------------- | ---- |
| Przystalski,  | K.  | et  al.  (2026)  | ‘Stylometry  |     | Recognizes  |               |         |      |             |         |                  |      |
|               |     |                  |              |     |             | empires,      | Nature  | and  | landscape,  |         | Neurodiversity,  |      |
Human and LLM-Generated Texts in Short Samples’,
Neuroscience, Oceans and water, Palaeontology, Personality,
Expert Systems with Applications, 296: 129001.
Philosophy of language, Philosophy of mind, Philosophy of
Ross, G. (2020) ‘Tracking the Evolution of Literary Style
religion, Philosophy of science, Physics, Pleasure and pain,
via Dirichlet–Multinomial Change Point Regression’,
Political philosophy, Politics and government, Poverty and
| Journal  | of  the  | Royal  | Statistical  | Society  | Series  A:  |               |           |     |                  |     |         |          |
| -------- | -------- | ------ | ------------ | -------- | ----------- | ------------- | --------- | --- | ---------------- | --- | ------- | -------- |
|          |          |        |              |          |             | development,  | Progress  |     | and  modernity,  |     | Public  | health,  |
Statistics in Society, 183: 149–67.
|     |     |     |     |     |     | Quantum  | theory,  | Religion,  | Ritual  | and  | celebrations,  | Self-  |
| --- | --- | --- | --- | --- | --- | -------- | -------- | ---------- | ------- | ---- | -------------- | ------ |
Rudman, J. (2006) ‘Authorship Attribution: Statistical
improvement, Sleep and dreams, Social psychology, Space
| and  Computational  |     | Methods’,  |     | in  Brown,  | K.  (ed.),  |     |     |     |     |     |     |     |
| ------------------- | --- | ---------- | --- | ----------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
exploration, Spirituality, Sports and games, Stories and litera-
Encyclopedia of Language and Linguistics, 2nd edn,
ture, Subcultures, Teaching and learning, Technology and
vol. 1, pp. 611–17. Elsevier: Oxford.
the self, The ancient world, The environment, The future,
Stamatatos, E. (2009) ‘A Survey of Modern Authorship
Thinkers and theories, Travel, Values and beliefs, Virtues and
Attribution Methods’, Journal of the American Society
|     |     |     |     |     |     | vices,  War  | and  | peace,  | Wellbeing,  |     | Work,  | Psychiatry,  |
| --- | --- | --- | --- | --- | --- | ------------ | ---- | ------- | ----------- | --- | ------ | ------------ |
for Information Science and Technology, 60: 538–56.
Psychotherapy.
| Torgerson, W. S. (1952)  |     |     | ‘Multidimensional Scaling: I.  |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Theory and Method’, Psychometrika, 17: 401–19.
Tyo, J., Dhingra, B., and Lipton, Z. C. (2022) On the  Appendix B: performance
state of the art in authorship attribution and author-
results of classifiers
ship verification. arXiv preprint arXiv : 2209.06869.,
preprint: not peer reviewed. across studies
Wu, J. et al. (2025) ‘A Survey on LLM-Generated Text
Detection: Necessity, Methods, and Future Directions’,  The tables present the numerical results for Studies 1
Computational Linguistics, 51: 275–338. through 4 to further substantiate the respective findings.

16                                                                                                                                                                           Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00
Study 1: Aggregated versus  Table 7. Performance metrics for SVMs on full-length essays
and the first 200 words of each essay.
Individual Treatment of
Human-Written Essays in Authorship  Model Accuracy   Recall   Precision   F1 (%)
|     |     |     |     | (%) (%) | (%) |
| --- | --- | --- | --- | ------- | --- |
Attribution
|     |     |     | Whole Length | 99.26 99.17 | 99.35 99.26 |
| --- | --- | --- | ------------ | ----------- | ----------- |
Table 3. Classifier results (Delta, RF, SVMs) on the  200-word 88.72 88.12 89.22 88.64
Downloaded from https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041 by guest on 23 July 2026
combined dataset.
| Model Accuracy (%) | Recall (%) | Precision (%) F1 (%) |     |     |     |
| ------------------ | ---------- | -------------------- | --- | --- | --- |
Study 3: Impact of Training Data Size
| Delta 66.65 | 62.26 | 68.30 65.05 | on Classifier Performance |     |     |
| ----------- | ----- | ----------- | ------------------------- | --- | --- |
| RF 95.03    | 99.82 | 91.11 95.26 |                           |     |     |
| SVMs 81.94  | 82.70 | 81.52 82.08 |                           |     |     |
Table 8. Classifier results (Delta, RF, SVMs) with 10% of each
topic used as training data.
|     |     |     | Model Accuracy (%) | Recall (%) | Precision (%) F1 (%) |
| --- | --- | --- | ------------------ | ---------- | -------------------- |
Table 4. Classifier results (Delta, RF, SVMs) on the
separate dataset.
|                    |            |                      | Delta 90.58 | 92.49 | 89.49 90.84 |
| ------------------ | ---------- | -------------------- | ----------- | ----- | ----------- |
|                    |            |                      | RF 98.36    | 99.55 | 97.39 98.42 |
| Model Accuracy (%) | Recall (%) | Precision (%) F1 (%) |             |       |             |
|                    |            |                      | SVMs 76.42  | 69.26 | 81.75 74.60 |
| Delta 97.98        | 96.00      | 99.95 97.93          |             |       |             |
| RF 99.93           | 99.95      | 99.91 99.93          |             |       |             |
Table 9. Classifier results (Delta, RF, SVMs) with 20% of each
| SVMs 99.26 | 99.17 | 99.36 99.26 |     |     |     |
| ---------- | ----- | ----------- | --- | --- | --- |
topic used as training data.
|     |     |     | Model Accuracy (%) | Recall (%) | Precision (%) F1 (%) |
| --- | --- | --- | ------------------ | ---------- | -------------------- |
Study 2: Impact of Essay Length on
| Classification Performance |     |     | Delta 97.82 | 98.16 | 97.54 97.83 |
| -------------------------- | --- | --- | ----------- | ----- | ----------- |
|                            |     |     | RF 99.31    | 99.77 | 98.87 99.31 |
|                            |     |     | SVMs 79.08  | 77.93 | 80.09 78.87 |
Table 5. Performance metrics for Delta on full-length essays
and the first 200 words of each essay. Table 10. Classifier results (Delta, RF, SVMs) with 50% of each
topic used as training data.
| Model | Accuracy   Recall   | Precision   F1   |     |     |     |
| ----- | ------------------- | ---------------- | --- | --- | --- |
(%) (%) (%) (%) Model Accuracy (%) Recall (%) Precision (%) F1 (%)
Whole length 97.98 96.00 99.95 97.93 Delta 99.54 99.54 99.55 99.54
| 200-word | 90.59 97.00 | 86.00 91.16 | RF 99.82   | 100.00 | 99.64 99.82 |
| -------- | ----------- | ----------- | ---------- | ------ | ----------- |
|          |             |             | SVMs 81.32 | 82.34  | 80.70 81.48 |
Table 6. Performance metrics for RF on full-length essays and  Study 4: Assessing the Influence of
the first 200 words of each essay.
Topic Relevance on the Performance
| Model Accuracy (%) | Recall (%) | Precision (%) F1 (%) |     |     |     |
| ------------------ | ---------- | -------------------- | --- | --- | --- |
of Classifiers
| Whole   99.93 | 99.95 | 99.91 99.93 |     |     |     |
| ------------- | ----- | ----------- | --- | --- | --- |
Access the output at: https://github.com/xiongshizhao/
length
stylometric-analysis-human-and-chatgpt-texts/tree/main/
| 200-word 50.69 | 100.00 | 50.35 66.98 |     |     |     |
| -------------- | ------ | ----------- | --- | --- | --- |
results/study4/tables

Digital Scholarship in the Humanities, 2026, Vol, 00, Issue 00 17
Appendix C: code and · R scripts for data analysis and visualization.
·
Raw and processed data files.
data repository
Access the repository at: https://github.com/xiongshiz
In this appendix, we provide a link to the GitHub reposi-
hao/stylometric-analysis-human-and-chatgpt-texts
tory that contains all the source code and datasets used
in the development of this report. The repository
includes the following:
© The Author(s) 2026. Published by Oxford University Press on behalf of EADH.
This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/
), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.
Digital Scholarship in the Humanities, 2026, 00, 1–17
https://doi.org/10.1093/llc/fqag064
Full Paper
Downloaded
from
https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
by
guest
on
23
July
2026
