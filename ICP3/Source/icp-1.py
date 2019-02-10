class Employee:
    """
    A class representing employee
    """
    counter = 0
    salary_sum = 0

    def __init__(self, n, f, s, d):
        self.name = n
        self.family = f
        self.salary = s
        self.department = d
        Employee.counter += 1
        Employee.salary_sum = Employee.salary_sum + s

    def __del__(self):
        Employee.counter -= 1
        Employee.salary_sum -= self.salary

    def display_employee(self):
        print("name : ", self.name, ", family : ", self.family, ", Salary: ", self.salary, ", department: ", self.department)

    def get_average_salary(self):
        average_salary = self.salary_sum / self.counter
        print("The average salary of employee is ", average_salary)
        return


class Fulltime(Employee):

    def __init__(self, n, f, s, d):
        Employee.__init__(self, n, f, s, d)


if __name__ == "__main__":

    employee1 = Employee("Jack", "John", 5000, "IT")
    employee2 = Employee("Jason", "JJ", 6000, "IT")
    employee3 = Employee("Max", "Jin", 7600, "IT")
    employee4 = Fulltime("Zel", "Corn", 6700, "IT")

    employee1.display_employee()
    employee2.display_employee()
    employee3.display_employee()
    employee4.display_employee()
    Employee.get_average_salary(Employee)
