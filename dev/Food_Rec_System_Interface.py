from Food_Rec_System_Engine import *

print("To use, open up the nutrition_data.csv for reference.")
print("The system needs 2 inputs: a food from the csv, and the number of recommendations you would like back.\n")
print("To quit, type quit in the food response")
active = True
while active:
    food = input("Input the food: \n")
    if food == "quit":
        break
    number_recs = input(input("Input the number of recs: \n"))



    find_n_closest(food, number_recs)


