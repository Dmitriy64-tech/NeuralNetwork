using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Topology
    {
        public int InputCount { get;  }
        public int OutputCount { get; }
        public double LearningRate { get; }
        public List<int> HidenLayers { get; }


        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = InputCount;
            OutputCount = outputCount;
            HidenLayers = new List<int>();
            HidenLayers.AddRange(layers);
            LearningRate = learningRate;
        }
    }
}
