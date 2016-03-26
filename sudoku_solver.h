#include<iostream>
using namespace std;
int possible(int a[9][9],int x,int y,int k)
{
	for(int i=0;i<9;i++)
		if(a[i][y]==k||a[x][i]==k)
			return 0;
	int x1=x/3,y1=y/3;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			if(a[i+x1*3][j+y1*3]==k)
				return 0;
	return 1;
}
int next(int a[9][9],int &x,int &y)
{
	for(int i=0;i<9;i++)
	{
		for(int j=0;j<9;j++)
		{
			if(a[i][j]==0)
			{    
				x=i;
				y=j;
				return 1;
			}
		}
	}
	return 0;
}
int sudoku(int a[9][9],int i,int j)
{
	for(int k=1;k<=9;k++)
	{
		if(possible(a,i,j,k))
		{
			a[i][j]=k;
			int x=i,y=j;
			if(next(a,x,y)==1)
			{
				if(sudoku(a,x,y)==1)
		      return 1;
			}
			else
				return 1;
		}
	}
	a[i][j]=0;
	return 0;
}
int solve(int a[9][9])
{
	int x=0,y=0;
	next(a,x,y);
	if(sudoku(a,x,y)==1)
		return 1;
	else
		return 0;
}
