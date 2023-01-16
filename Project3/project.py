# project.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns
    a dictionary with the following structure:
    The keys are the general areas of the syllabus: lab, project,
    midterm, final, disc, checkpoint
    The values are lists that contain the assignment names of that type.
    For example the lab assignments all have names of the form labXX where XX
    is a zero-padded two digit number. See the doctests for more details.
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    result = {}
    key = ['lab', 'project', 'Midterm','Final','disc','checkpoint']
    for i in range(len(key)):
        newkey = [x.lower() for x in key]
        temp1 = list(grades.filter(like = key[i], axis = 1).columns)
        res = np.array([])
        for j in temp1:
            temp2 = j.split('-')[0].replace(' ','')
            res = np.append(temp2, res)
        result[newkey[i]] = list(np.unique(res))
    new = []
    for i in result['project']:
        i = i.split('_')[0]
        new = np.append(new, i)
    result['project'] = list(np.unique((new)))
    return result


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus.
    The output Series should contain values between 0 and 1.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''

    df = grades.filter(like='project').fillna(0)
    select_columns = df.loc[:, ~df.columns.str.contains('Lateness|Max Points|checkpoint')]
    Max = df.filter(like='Max Points').fillna(0)
    max_columns = Max.loc[:, ~Max.columns.str.contains('checkpoint')]
    return select_columns.sum(axis=1) / max_columns.sum(axis=1)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe
    grades and returns a Series indexed by lab assignment that
    contains the number of submissions that were turned
    in on time by the student, yet marked 'late' by Gradescope.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """

    lab_lst = get_assignment_names(grades)["lab"]
    late_lab_lst = [x + " - Lateness (H:M:S)" for x in lab_lst]
    output_lst = []
    for i in late_lab_lst:
        grades["late_hour"] = grades[i].apply(lambda x: int(x.split(":")[0]))
        new_df = grades[grades[i] != "00:00:00"]
        # the threshhold should be eight hours
        output_lst.append(new_df[new_df["late_hour"].between(0, 7, inclusive=True)].shape[0])

    grades.drop(columns="late_hour", inplace = True)
    output_s = pd.Series(output_lst, lab_lst)
    return output_s


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    adjust_lateness takes in the dataframe like `grades`
    and returns a dataframe of lab grades adjusted for
    lateness according to the syllabus.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """
    new_s = col.apply(lambda x: int(x.split(":")[0]))

    def help_func(x):
        if x < 8:
            return 1.0
        if x < (24 * 7):
            return 0.9
        if x < (24 * 2 * 7):
            return 0.7
        else:
            return 0.4

    output_s = new_s.apply(help_func)
    return output_s



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment,
        adjusted for Lateness and scaled to a score between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    df2 = grades.filter(like='lab')

    temp = df2.loc[:, ~df2.columns.str.contains('Lateness|Max Points')]
    Max2 = df2.filter(like = 'Max Points')
    Max2.columns = temp.columns
    original_scores = np.divide(temp, Max2)

    lateness_columns = df2.loc[:, df2.columns.str.contains('Lateness')]
    dat = pd.DataFrame()

    for i in list(lateness_columns.columns):
        dat[i] = lateness_penalty(lateness_columns[i])

    dat.columns = temp.columns
    return np.multiply(dat,original_scores)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series).

    Your answers should be proportions between 0 and 1.
    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    func = lambda x : x.sort_values()[1:]
    output = processed.fillna(0).apply(func, axis = 1).mean(axis = 1)
    return output


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    lab = lab_total(process_labs(grades))
    project = projects_total(grades)

    def func(df, parm):
        temp = df.filter(like = parm).fillna(0)
        points = temp.loc[:, ~temp.columns.str.contains('Lateness|Max Points')]
        Max = temp.loc[:, temp.columns.str.contains('Max Points')]
        Max.columns = points.columns
        original_scores = np.divide(points, Max).sum(axis = 1)
        return original_scores / len(points.columns)

    checkpoint = func(grades, "checkpoint")
    discussion = func(grades, "discussion")
    midterm = func(grades, "Midterm")
    final = func(grades, "Final")

    return lab * 0.2 + project * 0.3 + checkpoint * 0.025 + discussion * 0.025 + midterm * 0.15 + final * 0.3

def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.
    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def help_func(x):
        if x >= 0.9:
            return "A"
        elif 0.9 > x >= 0.8:
            return "B"
        elif 0.8 > x >= 0.7:
            return "C"
        elif 0.7 > x >= 0.6:
            return "D"
        elif 0.6 > x:
            return "F"


    return total.apply(help_func)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades
    and outputs a Series that contains the proportion
    of the class that received each grade.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    scores = total_points(grades)
    letter_grade = final_grades(scores)
    result = (letter_grade.value_counts() / letter_grade.shape[0]).sort_values(ascending = False)
    return result


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 1000)
    >>> 0 <= out <= 0.1
    True
    """
    observed_stats = total_points(grades[grades['Level'] == 'SR']).mean()
    siz = len(grades[grades['Level'] == 'SR'])
    score_distr = total_points(grades).value_counts(normalize=True)
    samps = np.random.choice(score_distr.index,p=score_distr,size=(N, siz),replace = True)
    averages = samps.mean(axis=1)
    return (observed_stats >= averages).mean()


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades,
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    def func(grades):
        result = {}
        key = ['lab', 'project', 'Midterm', 'Final', 'disc', 'checkpoint']
        for i in range(len(key)):
            newkey = [x.lower() for x in key]
            temp1 = list(filter(lambda x: "checkpoint" not in x, list(grades.filter(like=key[i], axis=1).columns)))
            res = np.array([])
            for j in temp1:
                temp2 = j.split('-')[0].replace(' ', '')
                res = np.append(temp2, res)
            result[newkey[i]] = list(np.unique(res))
        contain_checkpoint = list(grades.filter(like="checkpoint", axis=1).columns)
        check_arr = np.array([])
        for a in contain_checkpoint:
            splitted = a.split('-')[0].replace(' ', '')
            check_arr = np.append(splitted, check_arr)
        result["checkpoint"] = list(np.unique(check_arr))
        return result


    new = grades.fillna(0)
    newdf = process_labs(new)
    assignment = pd.Series(func(new)).sum()

    noise = pd.DataFrame(np.random.normal(0, 0.02, size = (new.shape[0], len(assignment)))).clip(-1,1)
    noise.columns = assignment
    for i in assignment:

        if 'lab' in i:
            new[i] = ((newdf[i] + noise[i]) * new[i + ' - Max Points'])/ lateness_penalty(new[i + " - Lateness (H:M:S)"])
        else:
            new[i] = ((new[i] /new[i + ' - Max Points']) + noise[i]) * new[i + ' - Max Points']

    output = total_points(new).clip(0,1)
    return output


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def short_answer():
    """
    short_answer returns (hard-coded) answers to the
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.
    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """
    return [-0.000122514370581972, 0.8336448598130841,[80.364485, 86.55140],
        0.0616822429906542,[True, False]]
