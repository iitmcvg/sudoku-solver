#include<iostream>
using namespace std;
int possible(int a[9][9],int x,int y,int k)
{
	int i,j;
	for(i = 0 ; i < 9 ; i++) {
		if(a[i][y]==k || a[x][i]==k)
			return 0;
	}
	int x1 = x/3, y1 = y/3;
	for(i = 0 ; i < 3 ; i++) 
	{
		for(j = 0 ; j < 3 ; j++)
		{
			if(a[i+3*x1][j+3*y1] == k)
				return 0;
		}
	}
  return 1;
}
int next(int a[9][9],int &x,int &y)
{
	int i,j;
	for(i = 0 ; i < 9 ; i++) 
	{
		for(j = 0 ; j < 9 ; j++) 
		{
			if(a[i][j] == 0) 
			{
				x = i;
				y = j;
				return 1
			}
		}
	}
	return 0;
}
int check(int a[9][9])
{
	for(int i = 0 ; i < 9 ; i++)
	{
		for(int j = 0 ; j < 9 ; j++)
		{
			if(a[i][j] == 0)
				return 0;
		}
	}
	return 1;
}
int sudoku(int a[9][9],int i,int j)
{
	for(int k = 1 ; k <= 9 ; k++) 
	{
		if(possible(a,i,j,k))
		{
			a[i][j] = k;
			if(check(a) == 1)
				return 1;
			int x = i, y = j;
			if(next(a,x,y)) { 
				if(sudoku(a,x,y) == 1) return 1;
			}
		}
	}
	a[i][j] = 0;
	return 0;
}
int solve(int a[9][9])
{
	int x=0,y=0;
	next(a,x,y);
	return sudoku(a,x,y);
}
