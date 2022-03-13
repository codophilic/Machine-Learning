import numpy as np 
#hypothesis function h(x)(price)=b+m*area--->b=slope,m=intercept

def gradient_descent(area,price):
    m=b=0 #starts with intial values as 0 and after iterations it wil take small steps to go for the global minima.
    iterations=1000 #no. of steps 
    n=len(area)#length of the data set
    learning_rate=0.001 #Take a smaller value and depending on the behaviour of the cost function(the values has to be decreased after each iterations) 
    #change the value of the learning rate 
    for i in range(iterations):
        price_predicted=b+m*area#-->hypothesis function
        
        #Cost Function it must decrease depnding on the value of learning rate
        cost_function=(1/(2*n))*sum([val**2 for val in (price-price_predicted)])
        
        partial_derivative_wrt_m=-(1/n)*sum(area*(price-price_predicted)) #--->d/dm(J(m)) partial derivative of the cost function wrt slope(thetha1)
        partial_derivative_wrt_b=-(1/n)*sum((price-price_predicted)) #--->d/db(J(b)) partial derivative of the cost function wrt intercept(thetha0)
        
        #Gradient Descent
        m=m-learning_rate*partial_derivative_wrt_m
        b=b-learning_rate*partial_derivative_wrt_b  
        
        print("m",m,"b",b,"cost function",cost_function,"iteration",i)

area=np.array([1,2,3,4,5])
#print(0*area)--->multiplies all the array of the element by 0
price=np.array([5,7,9,11,13])
gradient_descent(area,price)