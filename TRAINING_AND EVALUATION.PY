import random
import glob
from csv import DictReader
import numpy as np
Q=LinearClassifier(0.001,1,3)
eps_min=0.01 
eps_dec=5e-7
epsilon=1
it=5
LOSS=[]
for j in range(1000):
 
  testL=[]
  LOSED=[]
  path = r'/content/training' # use your path
  all_files = glob.glob(path + "/*.csv")
  for filename in all_files:
    with open(filename, 'r') as read_obj:
     
      csv_dict_reader = DictReader(read_obj)
      i=0
      LOSED=[]
      
      batting.clear()
      bowling.clear()
      ground.clear()
      
      for row in csv_dict_reader:
        
 
        if(row['bowling']=='Rising Pune Supergiant' or row['batting']=='Rising Pune Supergiant'or row['batting']=='Rising Pune Supergiants' or row['bowling']=='Rising Pune Supergiants' or row['batting']=='Pune Warriors' or row['bowling']=='Pune Warriors' or row['batting']=='Gujarat Lions' or row['bowling']=='Gujarat Lions' or row['bowling']=='Kochi Tuskers Kerala' or row['batting']=='Kochi Tuskers Kerala'):
          continue
        if(row['batting'] in batting):
          #print(j)
          batting[row['batting']].append(float(row['total']))
        else:
          batting[row['batting']]=[]
          batting[row['batting']].append(float(row['total']))
        if((np.size(batting[row['batting']])>it)):
            u=batting[row['batting']][-2]
            v=np.min(batting[row['batting']][-6:-1])
            w=np.max(batting[row['batting']][-6:-1])
            #v=((np.array(ema(batting[row['batting']][-5:],3)))[-2:-1])
            z=batting[row['batting']][-4]
            p=batting[row['batting']][-3]
            #x=batting[row['batting']][-5]
            #y=(batting[row['batting']][-6])
          
        if((row['ground']) in ground): 
           ground[row['ground']].append(float(row['total']))
        else:
          ground[row['ground']]=[]
          ground[row['ground']].append(float(row['total']))
        if(np.size(ground[row['ground']])>3):
            
            xg=np.mean(ground[row['ground']][-5:-1])
            yg=np.min(ground[row['ground']][-5:-1])
            #zg=np.max(ground[row['ground']][:-1])
           
            
        
        else:
            m1=np.array(ground[row['ground']]).astype(np.float)
            xg=42
            yg=42
            #zg=xg
        if(row['bowling'] in bowling):
          bowling[row['bowling']].append(float(row['total']))
        else:
          bowling[row['bowling']]=[]
          bowling[row['bowling']].append(float(row['total']))
        if((np.size(bowling[row['bowling']])>it)):
            u1=bowling[row['bowling']][-2]
            v1=np.min(bowling[row['bowling']][-6:-1])
            #w1=np.max(bowling[row['bowling']][-6:-1])
            #v1=(np.array(ema(batting[row['bowling']][-5:],3)))[-2:-1]
            #y1=(bowling[row['bowling']][-6])
            z1=bowling[row['bowling']][-4]
            p1=bowling[row['bowling']][-3]
            #x1=bowling[row['bowling']][-5]
        else:
          continue    
        ins=batting[row['batting']][-6:-1]  
        a,a1,b1,angle=make(ins)
        ans1=pt(a,a1,b1,angle)
    
        
        ins=bowling[row['bowling']][-6:-1]  
        a,a1,b1,angle=make(ins)
        ans2=pt(a,a1,b1,angle)
        t=u
        if(t<=30):
          t=30
        elif(30<t<=33):
          t=33  
        elif(33<t<=36):
          t=36
        elif(36<t<=39):
          t=39
        elif(39<t<=42):
          t=42
        elif(42<t<=45):
          t=45
        elif(45<t<=48):
          t=48
        elif(48<t<=52):
          t=51
        elif(51<t<=54):
          t=54
        elif(54<t<=57):
          t=57
        elif(t>57):
          t=60    

        t1=u1
        if(t1<=30):
          t1=30
        elif(30<t1<=33):
          t1=33  
        elif(33<t1<=36):
          t1=36
        elif(36<t1<=39):
          t1=39
        elif(39<t1<=42):
          t1=42
        elif(42<t1<=45):
          t1=45
        elif(45<t1<=48):
          t1=48
        elif(48<t1<=52):
          t1=51
        elif(51<t1<=54):
          t1=54
        elif(54<t1<=57):
          t1=57
        elif(t1>57):
          t1=60 
        st=str(int(t))+str(int(t1))+str(int(float(row['wicket'])))
        if str not in dict:
          continue
        data=dict[st]
        t = data[0:len(data)-1].index(min(data[0:len(data)-1]))
        if(t==0):
          inp=random.randint(25,30)
        if(t==1):
          inp=random.randint(30,35)
        if(t==2):
          inp=random.randint(35,40)
        if(t==3):
          inp=random.randint(40,45)
        if(t==4):
          inp=random.randint(45,50)
        if(t==5):
          inp=random.randint(50,55)
        if(t==6):
          inp=random.randint(55,60)
              
        a=[ans1,ans2,float(row['wicket'])]
        
        
        Q.optimizer.zero_grad()
        a=T.tensor(a, dtype=T.float).to(Q.device)
        ans=Q.forward(a).float()
        total=float(row['total'])
       
        targets_flat = T.Tensor([total])
        loss = Q.loss(ans,targets_flat).to(Q.device)
       
        #print(loss,ans,total)
        loss.backward()
        
        Q.optimizer.step()
        LOSED.append(float(loss))
        
  print(np.mean(LOSED))
  with open('/content/test.csv', 'r') as read_obj:
     
      csv_dict_reader = DictReader(read_obj)
      i=0
      t=0
      mloss1=[]
      ltest=100
      for row in csv_dict_reader:
        
 
        if(row['bowling']=='Rising Pune Supergiant' or row['batting']=='Rising Pune Supergiant'or row['batting']=='Rising Pune Supergiants' or row['bowling']=='Rising Pune Supergiants' or row['batting']=='Pune Warriors' or row['bowling']=='Pune Warriors' or row['batting']=='Gujarat Lions' or row['bowling']=='Gujarat Lions' or row['bowling']=='Kochi Tuskers Kerala' or row['batting']=='Kochi Tuskers Kerala'):
          continue
        if(row['batting'] in batting):
          #print(j)
          batting[row['batting']].append(float(row['total']))
        else:
          batting[row['batting']]=[]
          batting[row['batting']].append(float(row['total']))
        if((np.size(batting[row['batting']])>it)):
            u=batting[row['batting']][-2]
            v=np.min(batting[row['batting']][-6:-1])
            w=np.max(batting[row['batting']][-6:-1])
            #v=((np.array(ema(batting[row['batting']][-5:],3)))[-2:-1])
            z=batting[row['batting']][-4]
            p=batting[row['batting']][-3]
            x=batting[row['batting']][-5]
            y=(batting[row['batting']][-6])
          
        if((row['ground']) in ground): 
           ground[row['ground']].append(float(row['total']))
        else:
          ground[row['ground']]=[]
          ground[row['ground']].append(float(row['total']))
        if(np.size(ground[row['ground']])>3):
            
            xg=np.mean(ground[row['ground']][-5:-1])
            yg=np.min(ground[row['ground']][-5:-1])
            #zg=np.max(ground[row['ground']][:-1])
           
            
        
        else:
            m1=np.array(ground[row['ground']]).astype(np.float)
            xg=42
            yg=42
            #zg=xg
        if(row['bowling'] in bowling):
          bowling[row['bowling']].append(float(row['total']))
        else:
          bowling[row['bowling']]=[]
          bowling[row['bowling']].append(float(row['total']))
        if((np.size(bowling[row['bowling']])>it)):
            u1=bowling[row['bowling']][-2]
            v1=np.min(bowling[row['bowling']][-6:-1])
            w1=np.max(bowling[row['bowling']][-6:-1])
            #v1=(np.array(ema(batting[row['bowling']][-5:],3)))[-2:-1]
            y1=(bowling[row['bowling']][-6])
            z1=bowling[row['bowling']][-4]
            p1=bowling[row['bowling']][-3]
            x1=bowling[row['bowling']][-5]
        else:
          continue    
        inp=batting[row['batting']][-6:-1]  
        a,a1,b1,angle=make(inp)
        ans1=pt(a,a1,b1,angle)
       
    
        inp2=bowling[row['bowling']][-6:-1]  
        a,a1,b1,angle=make(inp2)
        ans2=pt(a,a1,b1,angle)
        t=u
        if(t<=30):
          t=30
        elif(30<t<=33):
          t=33  
        elif(33<t<=36):
          t=36
        elif(36<t<=39):
          t=39
        elif(39<t<=42):
          t=42
        elif(42<t<=45):
          t=45
        elif(45<t<=48):
          t=48
        elif(48<t<=52):
          t=51
        elif(51<t<=54):
          t=54
        elif(54<t<=57):
          t=57
        elif(t>57):
          t=60    

        t1=u1
        if(t1<=30):
          t1=30
        elif(30<t1<=33):
          t1=33  
        elif(33<t1<=36):
          t1=36
        elif(36<t1<=39):
          t1=39
        elif(39<t1<=42):
          t1=42
        elif(42<t1<=45):
          t1=45
        elif(45<t1<=48):
          t1=48
        elif(48<t1<=52):
          t1=51
        elif(51<t1<=54):
          t1=54
        elif(54<t1<=57):
          t1=57
        elif(t1>57):
          t1=60 
        st=str(int(t))+str(int(t1))+str(int(float(row['wicket'])))
        if str not in dict:
          continue
        data=dict[st]
        t = data[0:len(data)-1].index(min(data[0:len(data)-1]))
        if(t==0):
          inp=random.randint(25,30)
        if(t==1):
          inp=random.randint(30,35)
        if(t==2):
          inp=random.randint(35,40)
        if(t==3):
          inp=random.randint(40,45)
        if(t==4):
          inp=random.randint(45,50)
        if(t==5):
          inp=random.randint(50,55)
        if(t==6):
          inp=random.randint(55,60)          
        a=[ans1,ans2,float(row['wicket'])]
        
       
        
        Q.optimizer.zero_grad()
        a=T.tensor(a, dtype=T.float).to(Q.device)
        ans=Q.forward(a).float()
        total=float(row['total'])
        targets_flat = T.Tensor([total])
        loss = Q.loss(ans,targets_flat).to(Q.device)
        mloss1.append(float(loss))
        
        if(loss>10):
          t+=1
        
      print(np.mean(mloss1),t,j)
      if((np.mean(testL))<ltest):
          EPOCH = j
          PATH = "/content/drive/MyDrive/dataset/model.pt41"
          LOSS = loss

          T.save({
            'epoch': EPOCH,
            'model_state_dict': Q.state_dict(),
            'optimizer_state_dict': Q.optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
          ltest=np.mean(testL)
           
