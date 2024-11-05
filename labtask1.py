with open("students_info.txt", "r") as file:
    for line in file:
        student_data = line.strip().split(",")
        print(student_data)


unselected_student = []

selected_students = []

with open("students_info.txt", "r") as file:
    for line in file:
        student_data = line.strip().split(", ")

        if student_data[-1] == "No":
            unselected_student.append(student_data)

print("number of unselected student  : ", len(unselected_student))
print("List of unselected students:")
for student in unselected_student:
    print(student)

import random


def select_random_student():

     selected_student = random.choice(unselected_student)
     selected_student[-1] = "Yes"
     selected_students.append(selected_student.copy())
     unselected_student.remove(selected_student)
     return selected_student

for i in range (13):
  select_random_student()


print("number of unselected student  : ", len(unselected_student))
print("List of unselected students:")
for student in unselected_student:
    print(student)


print("number of unselected student  : ", len(selected_students))
print("List of selected students:")
for student in selected_students:
    print(student)





        



