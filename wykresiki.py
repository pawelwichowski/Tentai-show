import numpy as np
import random
import sys
import matplotlib.pyplot as plt
### FUNKCJE POMOCNICZE ###

#obliczanie komórki symetrycznej
def sym_point(center,cell):
    cx,cy=center
    x,y=cell
    return (2*cx-x,2*cy-y)


#sprawdzanie czy przypisanie do galaktyki jest poprawne
def is_valid_symmetric(board,center,cells):
    for cell in cells:
        sym_cell=sym_point(center,cell)
        if sym_cell not in cells:
            return False
        #sprawdzanie czy znajduje się w granicach plansy
        sx, sy = sym_cell
        if not (0 <= sx < board.shape[0] and 0 <= sy < board.shape[1]):
            return False
    return True

#tworzenie tablicy gęstości dla algorytmów density
def generate_density(board,centers):
    #dystans = |xc-x|+|xc-y| brak poruszania się "na ukos"
    #im mniejsza liczba tym bliżej innych centrów jest komórka
    n, m = board.shape
    solution = [[0 for j in range(m)]for i in range(n)]# wartość początkowa
    
    for center in centers:
        x,y=center
        for i in range(n):
            for j in range(m):
                #zapisujemy dystans oraz najbliższe centrum
                distance=abs(i-x)+abs(j-y)
                if solution[i][j] is 0:
                    solution[i][j]=[distance,distance,x,y]
                else:
                    #jeżeli dydtans do obecnego centrum jest mniejszy to zamaniętujemy yo
                    if distance<solution[i][j][1]:
                        solution[i][j]=[solution[i][j][0]+distance,distance,x,y]
                    else:
                        solution[i][j][0]+=distance
    
    #tworzenie listy centrów według największej gęstości
    wynik=[]
    for center in centers:
        x,y=center
        wynik.append((solution[x][y][0],x,y))
    wynik_max=sorted(wynik,key=lambda x:x[0])   
    wynik_min=sorted(wynik,key=lambda x:x[0],reverse=True) 
                        
                
                
    return solution,wynik_max,wynik_min

###########################
#       ZACHŁANNY        #
###########################


#główna funkcja rozwiązująca
def solve_greedy(board,centers):
    
    n, m = board.shape
    solution = np.full_like(board, -1)  # -1 oznacza nieprzypisaną komórkę
    
    for idx, center in enumerate(centers):
        #tworzymy kolejke do przegladania po koleji
        cx, cy = center
        queue = [(cx, cy)]
        #zbiór komórek należących do danej galaktyki
        galaxy_cells = set(queue)
        
        while queue:
            x, y = queue.pop(0)
            
            #patrzymy sąsiednie komórki obecnie dodanej
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                tmp=set(galaxy_cells)
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in galaxy_cells:
                    symmetric_cell = sym_point((cx, cy), (nx, ny))
                    sx, sy = symmetric_cell
                    #jeśli nie są przypisane do innej galaktyki to dodajemy do obecnej
                    tmp.add(symmetric_cell)
                    tmp.add((nx, ny))
                    if (0 <= sx < n and 0 <= sy < m and solution[nx, ny] == -1 and solution[sx, sy] == -1 and is_valid_symmetric(board,center,tmp)):
                        queue.append((nx, ny))
                        galaxy_cells.add((nx, ny))
                        galaxy_cells.add(symmetric_cell)
        
        if is_valid_symmetric(board, (cx, cy), galaxy_cells):
            for cell in galaxy_cells:
                x, y = cell
                solution[x, y] = idx
        else:
            raise ValueError("Nie można znaleźć rozwiązania")
    
    return solution
    
#####################################
#       ZACHŁANNY ULOSOWIONY       #
#####################################
    
def solve_greedy_random(board,centers):

    n, m = board.shape
    solution = np.full_like(board, -1)  # -1 oznacza nieprzypisaną komórkę
    random.shuffle(centers)
    for idx, center in enumerate(centers):
        cx, cy = center
        queue = [(cx, cy)]
        galaxy_cells = set(queue)

        while queue:
            # Losowy wybór komórki z kolejki
            random.shuffle(queue)
            x, y = queue.pop(0)

            # Losowy porządek sąsiadów
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(neighbors)

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in galaxy_cells:
                    symmetric_cell = sym_point((cx, cy), (nx, ny))
                    sx, sy = symmetric_cell
                    if (0 <= sx < n and 0 <= sy < m and 
                        solution[nx, ny] == -1 and solution[sx, sy] == -1):
                        queue.append((nx, ny))
                        galaxy_cells.add((nx, ny))
                        galaxy_cells.add(symmetric_cell)

        if is_valid_symmetric(board, (cx, cy), galaxy_cells):
            for cell in galaxy_cells:
                x, y = cell
                solution[x, y] = idx
        else:
            raise ValueError("Nie można znaleźć rozwiązania")

    return solution

##################################
#       NEAREST_NEIGHBOUR        #
##################################

def solve_nearest_neighbour(board,centers):
    
    n, m = board.shape
    solution = np.full_like(board, -1)  # -1 oznacza nieprzypisaną komórkę
    
    #iterujemy po kolejnych możliwych odległościach od centrum
    galaxies=[]
    for idx,center in enumerate(centers):
        galaxies.append([center])
        x,y=center
        solution[x,y]=idx
        
    for i in range(max(n,m)):

        for idx,galaxy in enumerate(galaxies):

            galaxy_cells=set()
            center=galaxy[0]
            cx, cy = center
            for cell in galaxy:
                x, y = cell
                
                #patrzymy sąsiednie komórki obecnie dodanej
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in galaxy:
                        symmetric_cell = sym_point((cx, cy), (nx, ny))
                        sx, sy = symmetric_cell
                        #jeśli nie są przypisane do innej galaktyki to dodajemy do obecnej
                        if (0 <= sx < n and 0 <= sy < m and solution[nx, ny] == -1 and solution[sx, sy] == -1):
                            galaxy_cells.add((nx, ny))
                            galaxy_cells.add(symmetric_cell)

            if is_valid_symmetric(board, (cx, cy), galaxy_cells):
                for cell in galaxy_cells:
                    x, y = cell
                    solution[x, y] = idx
                    galaxies[idx].append(cell)
            else:
                raise ValueError("Nie można znaleźć rozwiązania")
        #print(solution)
    return solution

###########################
#       DENSITY MIN       #
###########################

def solve_density_min(board,centers):
    
    #generowanie tablicy gęstości oraz listy centrów względem gęstości
    density,centers_den,holder=generate_density(board,centers)
    
    
    n, m = board.shape
    solution = np.full_like(board, -1)  # -1 oznacza nieprzypisaną komórkę
    
    #wypełnianie center by program nie myslał że są puste
    for idx,center in enumerate(centers):
        x,y=center
        solution[x][y]=idx
    
    for idx, center in enumerate(centers_den):
        d,cx, cy = center
        queue = [(cx, cy)]
        galaxy_cells = set(queue)
        
        while queue:
            x, y = queue.pop(0)
            adjecant=[(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            #sortowanie adjecant po najmniejszej gęstości
            tmp=[]
            for dx,dy in adjecant:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    tmp.append((density[nx][ny][0],dx,dy))
            tmp.sort(key=lambda x:x[0],reverse=True)

            new_adjecant=[]
            for i in tmp:
                new_adjecant.append((i[1],i[2]))
            
            #uzupełnianie by pasowało dalej
            for e in adjecant:
                if e not in new_adjecant:
                    new_adjecant.append(e)
            

            for dx, dy in new_adjecant:
                tmp=set(galaxy_cells)
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in galaxy_cells:
                    symmetric_cell = sym_point((cx, cy), (nx, ny))
                    sx, sy = symmetric_cell
                    tmp.add(symmetric_cell)
                    tmp.add((nx, ny))
                    if (0 <= sx < n and 0 <= sy < m and solution[nx, ny] == -1 and solution[sx, sy] == -1 and is_valid_symmetric(board,(cx,cy),tmp)):
                        queue.append((nx, ny))
                        galaxy_cells.add((nx, ny))
                        galaxy_cells.add(symmetric_cell)
                        
        if is_valid_symmetric(board, (cx, cy), galaxy_cells):
            for cell in galaxy_cells:
                x, y = cell
                solution[x, y] = idx
        else:
            raise ValueError("Nie można znaleźć rozwiązania")
    
    return solution

###########################
#       DENSITY MAX       #
###########################

def solve_density_max(board,centers):
    
    #generowanie tablicy gęstości oraz listy centrów względem gęstości
    density,holder,centers_den=generate_density(board,centers)
    
    
    n, m = board.shape
    solution = np.full_like(board, -1)  # -1 oznacza nieprzypisaną komórkę
    
    #wypełnianie center by program nie myslał że są puste
    for idx,center in enumerate(centers):
        x,y=center
        solution[x][y]=idx
    
    for idx, center in enumerate(centers_den):
        d,cx, cy = center
        queue = [(cx, cy)]
        galaxy_cells = set(queue)
        
        while queue:
            x, y = queue.pop(0)
            adjecant=[(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            #sortowanie adjecant po najwięszej gęstości
            tmp=[]
            for dx,dy in adjecant:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    tmp.append((density[nx][ny][0],dx,dy))
            tmp.sort(key=lambda x:x[0])

            new_adjecant=[]
            for i in tmp:
                new_adjecant.append((i[1],i[2]))
            
            #uzupełnianie by pasowało dalej
            for e in adjecant:
                if e not in new_adjecant:
                    new_adjecant.append(e)
            

            for dx, dy in new_adjecant:
                tmp=set(galaxy_cells)
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in galaxy_cells:
                    symmetric_cell = sym_point((cx, cy), (nx, ny))
                    sx, sy = symmetric_cell
                    tmp.add(symmetric_cell)
                    tmp.add((nx, ny))
                    if (0 <= sx < n and 0 <= sy < m and solution[nx, ny] == -1 and solution[sx, sy] == -1 and is_valid_symmetric(board,(cx,cy),tmp)):
                        queue.append((nx, ny))
                        galaxy_cells.add((nx, ny))
                        galaxy_cells.add(symmetric_cell)
                        
        if is_valid_symmetric(board, (cx, cy), galaxy_cells):
            for cell in galaxy_cells:
                x, y = cell
                solution[x, y] = idx
        else:
            raise ValueError("Nie można znaleźć rozwiązania")
    
    return solution

def show_centers(board,centers):
    solution = np.full_like(board, -1)
    for idx,center in enumerate(centers):
        x,y=center
        solution[x,y]=idx
    return solution

def test_quality(board):
    n=0
    for row in board:
        for e in row:
            if e==-1:
                n+=1
    return n
        
if __name__ == "__main__":
    # ustawienia domyslne
    if len(sys.argv)==1:
        n=50 #rozmiar
        c=20 #ilość centrów
        p=20 #ilość prób
        
    elif len(sys.argv)!=4:
        print("Proszę podać 3 parametry: rozmiar, ilość centrów, ilość prób \n")
        sys.exit(1)
    
    else:
        n=int(sys.argv[1])
        c=int(sys.argv[2])
        p=int(sys.argv[3])
    
    wynik=[0,0,0,0,0]
    for i in range(p):
        
        test=np.zeros((n,n),dtype=int)
        #liczba centrów
        
        centers=[]
        for j in range(c):
            x=random.randint(0,n-1)
            y=random.randint(0,n-1)
            if (x,y) not in centers:
                centers.append((x,y))

        solution=solve_greedy(test,centers)
        wynik[0]+=test_quality(solution)

        solution=solve_greedy_random(test,centers)
        wynik[1]+=test_quality(solution)

        solution=solve_nearest_neighbour(test,centers)
        wynik[2]+=test_quality(solution)

        solution=solve_density_min(test,centers)
        wynik[3]+=test_quality(solution)

        solution=solve_density_max(test,centers)
        wynik[4]+=test_quality(solution)
        
    for i in wynik:
        i=i/p
    
    print(wynik)
    
    labels=["Zachłanny","Zachłanny\nulosowiony","Najbliższy\nsąsiad","Gęstość\nnajmniejsza","Gęstość\nnajwiększa"]    
    plt.bar(labels,wynik)
    plt.title("Średnia liczba niewypełnionych komórek")
    plt.ylabel("Ilość niewypełnionych komórek")
    
    plt.show()    
