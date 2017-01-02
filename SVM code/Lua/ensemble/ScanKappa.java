import java.util.Arrays;
import java.util.ArrayList;
import java.util.Comparator;
import java.io.*;
import java.util.*;

public class ScanKappa
{
    public static int[] order(float[] elemements)
    {
        return order(elemements, true);
    }

    public static int[] order(float[] elemements, boolean increase)
    {
        int[] rank = new int[elemements.length];
        order(elemements, increase, rank);
        return rank;
    }

    public static void order(float[] elemements, boolean increase, int[] rank)
    {
        class tmp_sort
        {
            public float value;
            public int index;
        }

        tmp_sort[] result = new tmp_sort[elemements.length];

        for (int i = 0; i < elemements.length; i++)
        {
            result[i] = new tmp_sort();
            result[i].value = elemements[i];
            result[i].index = i;
        }

        if (increase)
            Arrays.sort(result, new Comparator<tmp_sort>()
        {
            public int compare(tmp_sort tmp_sort, tmp_sort tmp_sort1)
            {
                if (tmp_sort.value < tmp_sort1.value)
                    return -1;
                if (tmp_sort.value > tmp_sort1.value)
                    return 1;
                else
                    return 0;
            }
        });
        else
            Arrays.sort(result, new Comparator<tmp_sort>()
        {
            public int compare(tmp_sort tmp_sort, tmp_sort tmp_sort1)
            {
                if (tmp_sort.value < tmp_sort1.value)
                    return 1;
                if (tmp_sort.value > tmp_sort1.value)
                    return -1;
                else
                    return 0;
            }
        });

        for (int i = 0; i < result.length; i++)
            rank[i] = result[i].index;

    }

    public static List<String> readList(String file)
    {
        List<String> result = new ArrayList<String>();
        try
        {
            BufferedReader rd = new BufferedReader(new FileReader(file));
            String line = null;
            while ((line = rd.readLine()) != null)
				result.add(line);

        }
        catch (IOException e)
        {
            e.printStackTrace();
            System.err.println("error");
            System.exit(1);
        }
        return result;
    }

    static void toArray(List<String> lines, float [] score, int [] target)
    {
        int i = 0;
        for (String l : lines)
        {
            String [] f = l.split("\t");
            score[i] = Float.parseFloat(f[0]);
            target[i] = Integer.parseInt(f[1]);
			i++;
        }
    }

	static int[][] findSeedOffset(float [] sorted_score, float [] seed, int[] offset, float eps)
	{
		int [][] result = new int[seed.length][2];
		int j = 0;
		int findCnt = 0;
		for (int i = 0 ; i < sorted_score.length; i++)
		{
			if (sorted_score[i] + eps > seed[j])
			{
				int s = i;
				int k = i;
				for (k = i ; k < sorted_score.length; k++)
				{
					if (sorted_score[k] > seed[j])
						break;
				}
				s -= offset[j];
				int e = k+ offset[j];
				i = k-1;
				result[j][0] = Math.max(s, 0);
				result[j][1] = Math.min(e, sorted_score.length-1);
				j++;
			}
			if (j == seed.length) break;
		}
		if (j < seed.length)
		{
			System.err.println("not all the seed find anchor");
			System.exit(1);
		}
		return result;
	}

    public static int [] scanKappa(float [] sorted_score,  int [] sorted_target, int [][] ends, float [] best)
    {
        int N = 5;
        float [] targetCnt = new float[N];
        Arrays.fill(targetCnt, 0);
        for (int i : sorted_target)
            targetCnt[i]++;

        float [][] W = new float[N][N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
			{
                W[i][j] = (i-j)*(i-j);
				W[i][j] /= (N-1)*(N-1);
			}

        int total = sorted_target.length;
        for (int i = 0; i < N; i++)
            targetCnt[i] /= total;

        int [][] countMat = new int[total+1][N];
		for (int i = 0; i < total + 1; i++)
			Arrays.fill(countMat[i], 0);

        for (int i = 0; i < total; i++ )
        {
            int tgt = sorted_target[i];
            for (int  j = 0; j < N; j++)
            {
                if (tgt == j)
                    countMat[i+1][j] = countMat[i][j] + 1;
                else
                    countMat[i+1][j] = countMat[i][j];
            }
        }

        float bestKappa = -10000;
        int [] bestCut = new int[N-1];

        float [] w4 = W[3];
        float [] w5 = W[4];
		float [] tgt = targetCnt;
		int [] finalCnt = countMat[total];

        for(int i = ends[0][0]; i < ends[0][1]; i+=2)
        {
			System.err.println("# " + i);
            int [] cnt1 = countMat[i];
            float [] w1 = W[0];
            float sum1 = 0;
            float E1 = 0;
            int c1 = i;
            for(int m = 0; m < N ; m++)
            {
                sum1 += cnt1[m]*w1[m];
                E1 += w1[m]*c1*tgt[m];
            }

            for (int j = Math.max(ends[1][0], i+1); j <= ends[1][1]; j++)
            {
                int [] cnt2 = countMat[j];
                float [] w2 = W[1];
                float sum2 = sum1;
                float E2 = E1 ;
                int c2 = j-i;
                for(int m = 0; m < N ; m++)
                {
                    sum2 += (cnt2[m]-cnt1[m])*w2[m];
                    E2 += w2[m]*c2*tgt[m];
                }

				for (int k = Math.max(ends[2][0], j+1); k <= ends[2][1]; k++)
                {
                    int [] cnt3 = countMat[k];
                    float [] w3 = W[2];
                    float sum3 = sum2;
                    float E3 = E2 ;
                    int c3 = k-j;
                    for(int m = 0; m < N ; m++)
                    {
                        sum3 += (cnt3[m]-cnt2[m])*w3[m];
                        E3 += w3[m]*c3*tgt[m];
                    }

					for (int l = Math.max(ends[3][0], k+1); l <= ends[3][1]; l++)
                    {
                        int [] cnt4 = countMat[l];
                        float sum4 = sum3;
                        float E4 = E3;
                        int c4 = l-k;
                        int c5 = total-l;
                        for(int m = 0; m < N ; m++)
                        {
                            sum4 += (cnt4[m]-cnt3[m])*w4[m] + (finalCnt[m]-cnt4[m])* w5[m];
                            E4 += (w4[m]*c4 + w5[m]*c5)*tgt[m];
                        }
                        float kappa = 1-sum4/E4;
                        if (kappa > bestKappa)
                        {
                            bestKappa = kappa;
                            bestCut[0] = i;
                            bestCut[1] = j;
                            bestCut[2] = k;
                            bestCut[3] = l;
                        }
                    }
                }
            }
        }
		best[0] = bestKappa;
		printMat(countMat, bestCut);
        return bestCut;
    }

    public static int [] scanKappa(float [] sorted_score,  int [] sorted_target, int start, float [] best)
    {
        int N = 5;
        float [] targetCnt = new float[N];
        Arrays.fill(targetCnt, 0);
        for (int i : sorted_target)
            targetCnt[i]++;

        float [][] W = new float[N][N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
			{
                W[i][j] = (i-j)*(i-j);
				W[i][j] /= (N-1)*(N-1);
			}

        int total = sorted_target.length;
        for (int i = 0; i < N; i++)
            targetCnt[i] /= total;

        int [][] countMat = new int[total+1][N];
		for (int i = 0; i < total + 1; i++)
			Arrays.fill(countMat[i], 0);

        for (int i = 0; i < total; i++ )
        {
            int tgt = sorted_target[i];
            for (int  j = 0; j < N; j++)
            {
                if (tgt == j)
                    countMat[i+1][j] = countMat[i][j] + 1;
                else
                    countMat[i+1][j] = countMat[i][j];
            }
        }

        int start1 = start;
        int end1  = total - 4;

        float bestKappa = -10000;
        int [] bestCut = new int[N-1];

        float [] w4 = W[3];
        float [] w5 = W[4];
		float [] tgt = targetCnt;
		int [] finalCnt = countMat[total];

        for(int i = start1; i < end1; i+=2)
        {
			System.err.println("# " + i);
            int [] cnt1 = countMat[i];
            float [] w1 = W[0];
            float sum1 = 0;
            float E1 = 0;
            int c1 = i;
            for(int m = 0; m < N ; m++)
            {
                sum1 += cnt1[m]*w1[m];
                E1 += w1[m]*c1*tgt[m];
            }

            for (int j = i + 1; j <= total; j++)
            {
                int [] cnt2 = countMat[j];
                float [] w2 = W[1];
                float sum2 = sum1;
                float E2 = E1 ;
                int c2 = j-i;
                for(int m = 0; m < N ; m++)
                {
                    sum2 += (cnt2[m]-cnt1[m])*w2[m];
                    E2 += w2[m]*c2*tgt[m];
                }

                for (int k = j + 1; k <= total; k++)
                {
                    int [] cnt3 = countMat[k];
                    float [] w3 = W[2];
                    float sum3 = sum2;
                    float E3 = E2 ;
                    int c3 = k-j;
                    for(int m = 0; m < N ; m++)
                    {
                        sum3 += (cnt3[m]-cnt2[m])*w3[m];
                        E3 += w3[m]*c3*tgt[m];
                    }

                    for (int l = k + 1; l <= total; l++)
                    {
                        int [] cnt4 = countMat[l];
                        float sum4 = sum3;
                        float E4 = E3;
                        int c4 = l-k;
                        int c5 = total-l;
                        for(int m = 0; m < N ; m++)
                        {
                            sum4 += (cnt4[m]-cnt3[m])*w4[m] + (finalCnt[m]-cnt4[m])* w5[m];
                            E4 += (w4[m]*c4 + w5[m]*c5)*tgt[m];
                        }
                        float kappa = 1-sum4/E4;
                        if (kappa > bestKappa)
                        {
                            bestKappa = kappa;
                            bestCut[0] = i;
                            bestCut[1] = j;
                            bestCut[2] = k;
                            bestCut[3] = l;
                        }
                    }
                }
            }
        }
		best[0] = bestKappa;
		printMat(countMat, bestCut);
        return bestCut;
    }

	static void printMat(int [][] countMat, int [] cut)
	{
		int lastInd = countMat.length-1;
		System.err.println ("{");
		for (int i = 0; i < cut.length+1 ; i++)
		{
			int j ;
			System.err.print("{");
			for (j = 0; j < countMat[0].length; j++)
			{
				int lastCnt = 0;
				int cnt;
				if (i > 0 )
					lastCnt = countMat[cut[i-1]][j];
				if (i == cut.length)
					cnt = countMat[lastInd][j];
				else
					cnt = countMat[cut[i]][j];
				if (j >0)
					System.err.print (",");
				System.err.print (cnt- lastCnt);
			}
			if (i == cut.length +1)
				System.err.println ("}");
			else
				System.err.println ("},");
		}
		System.err.println ("}");
	}

	static void sample(float [] score, int [] target, float pct, List<Float> subScore, List<Integer> subTarget)
	{
		Random rnd = new Random();
		for (int i = 0; i < score.length; i++)
		{
			float f = rnd.nextFloat();
			if (f < pct)
			{
				subScore.add(score[i]);
				subTarget.add(target[i]);
			}
		}
	}

    public static void main(String [] arg)
    {
        List<String> lines = readList(arg[0]);
        float [] score = new float[lines.size()];
        int [] target = new int[lines.size()];
        float [] sorted_score = new float[lines.size()];
        int [] sorted_target = new int[lines.size()];
        toArray(lines, score, target);
        int [] rank = order(score);
		int j = 0;
		int offset = 250; 
        for (int i : rank)
        {
            sorted_target[j] = target[i];
            sorted_score[j] = score[i];
			j++;
        }
		float [] bestScore = new float[1];
        int [] bestCuts = null;

		if (arg.length == 2) 
		{
			int start = Integer.parseInt(arg[1]);
			bestCuts = scanKappa(sorted_score, sorted_target, start, bestScore);
		}
		else if (arg.length == 3 || arg.length == 1) 
		{
			int start;
			float pct;
			if (arg.length==3)
			{
				start = Integer.parseInt(arg[1]);
				pct = Float.parseFloat(arg[2]);
			}
			else
			{
				start = (int)(sorted_score.length *0.6);
				pct = Math.min(1200f/sorted_score.length, 0.3f);
			}

			List<Float> subScore = new ArrayList<Float> ();
			List<Integer> subTarget = new ArrayList<Integer> ();
			sample(sorted_score, sorted_target, pct, subScore, subTarget);
			float [] score1 = new float[subScore.size()];
			for (int i = 0; i < score1.length; i++)
				score1[i] = subScore.get(i);
			int [] target1 = new int[subTarget.size()];
			for (int i = 0; i < score1.length; i++)
				target1[i] = subTarget.get(i);
			int start1 = (int)(start * pct);
			int [] seedCut = scanKappa(score1, target1, start1, bestScore);
			float [] seed = new float[seedCut.length];
			int [] offSet = new int[seedCut.length];
			//Arrays.fill(offSet, offset);
			offSet[0] = 150;
			offSet[1] = 120;
			offSet[2] = 120;
			offSet[3] = 120;
			for (int i = 0; i < seedCut.length; i++)
				seed[i] = score1[seedCut[i]-1];
			int [][] searchRange = findSeedOffset (sorted_score, seed, offSet, 0.05f);
			for (int i = 0; i < 4; i++)
				System.err.println(searchRange[i][0] + " " + searchRange[i][1]);

			bestCuts = scanKappa(sorted_score, sorted_target, searchRange, bestScore);
		}
		else
		{
			System.err.println("useage: java ScanKappa score_list start_position_to_search or java ScanKappa score_list start_position_to_search percent for presearch");
			System.exit(1);
		}
		System.err.println("best score is " + bestScore[0]);
		System.err.println("best cut off");
		for (int i = 0; i < bestCuts.length; i++)
			System.err.println(bestCuts[i] + " "+ sorted_score[bestCuts[i]-1]);

		System.out.print(arg[0] + "\t" + bestScore[0]);
		for (int i = 0; i < bestCuts.length; i++)
			System.out.print("\t" + bestCuts[i] + ":"+ sorted_score[bestCuts[i]-1]);
		System.out.println();
    }
}
