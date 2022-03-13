#Adding more spam mails

import pandas as pd

df={'Name':['A','B'],
    "roll":[1,2],
    'wt':[50,90]
    }
df=pd.DataFrame(df)

new={'Name':['c','d'],
     'roll':[3,4]
     #'wt':[70,80]
     }
new=pd.DataFrame(new)
df=df.append(new,ignore_index=True)

dataset = pd.read_csv('spam1.csv')

n=20
new_data={
    'Unnamed: 0':[i for i in range(5171,5171+n,1)],
    'label':['spam' for i in range(n)],
    'text':[
        'Congratulations! Affordable Insurance for Coronavirus | Upto INR 1 CR Health Coverage. Get INR 10 Lacs Health Coverage. Click Here For Details',
        'iROBOT - Post Your Resume. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER	 	I AM A PROFESSIONAL (Experienced)',
        'RE: INSTANTLY ACTIVATE NO CHARGE SAVINGS ACCOUNT TODAY. Click Here For Details',
        'TESLA MNC Hiring Today for 0 to 13 Yrs Experience. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER   	I AM A PROFESSIONAL (Experienced)',
        'APPLIED MATERIALS- Submit Your Resume. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER   	I AM A PROFESSIONAL (Experienced)',
        'TESLA MNC Hiring Today for 0 to 13 Yrs Experience. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER   	I AM A PROFESSIONAL (Experienced)',
        'Hello! You Are Invited to Apply for Standard Chartered DigiSmart credit card. Click Here For Details',
        'Registered Member Profile Shortlisted To Apply For Credit Card from State Bank Of India (SBI). Click Here For Details',
        'Congratulations! Affordable Insurance for Coronavirus | Upto INR 1 CR Health Coverage. Get INR 10 Lacs Health Coverage. Click Here For Details',
        'Congrats! NEW MEMBER Shortlisted To Apply For Credit Card from Citi Bank of India (CT53). Click Here For Details',
        'Congratulations! Affordable Insurance for Coronavirus | Upto INR 1 CR Health Coverage. Click Here For Details',
        'Hello! You Are Invited to Apply for Standard Chartered DigiSmart credit card. Click Here For Details',
        'ADOBE INDIA Recruitment 0 to 15 Yrs Exp | Salary 25k to 125k. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER   	I AM A PROFESSIONAL (Experienced)',
        'BMW - Submit Your Resume. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER   	I AM A PROFESSIONAL (Experienced)',
        'RELIANCE INDUSTRIES (RIL) RECRUITMENT - Submit Your Resume. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER   	I AM A PROFESSIONAL (Experienced)',
        'Membership Message | 1 paisa brokerage Hurry!. Click Here For Details',
        'HEWLETT-PACKARD - Submit Your Resume. Choose Your Experience Level Below And Submit Your Resume. I AM A FRESHER,  	I AM A PROFESSIONAL (Experienced)',
        'RE: GET INSTANT 811 SAVINGS ACCOUNT - Urgent 49. Click Here For Details.'
        'free 30-seconds smile assessment get started on the smile you love in just 3 steps use code SDCAFF50 for 50% off',
        'Get free Quote TruGreen offer',
        'Get credit card analysis less than 1 minute for free',
        ],
    'label_num':[1 for i in range(n)]
    }
    
new_data=pd.DataFrame(new_data)

dataset_modified=dataset.append(new_data,ignore_index=True)
#This is a new modified spam.csv 
dataset_modified.to_csv('spam_modified.csv',index=False)





import pandas as pd

dfm=pd.read_csv('spam_modified.csv')
    
    
    
    
    
    
    
    

















