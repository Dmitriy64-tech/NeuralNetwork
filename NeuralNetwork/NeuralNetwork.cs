using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }
        public List<Layer>  Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();
            CreateInputLayer();
            CreateHiddenLayers();
            CreateoutputLayer();
        }
        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();
            if (Topology.InputCount==1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);

                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];
                neuron.FeedForward(signal);
            }
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HidenLayers.Count; j++)
            {
                var HiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HidenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    HiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(HiddenNeurons);
                Layers.Add(hiddenLayer);
            }
            
        }

        private void CreateoutputLayer()
        {
            var OutputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                OutputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(OutputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateInputLayer()
        {
            var InputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                InputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(InputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error += BackPropagation(output, input);
                }
            }

            var result = error / epoch;
            return result;
        }
        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; ++i)
                array[i] = matrix[row, i];
            return array;
        }

        private double[,] Scalling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int colums = 0; colums < inputs.GetLength(1); colums++)
            {
                var min = inputs[0, colums];
                var max = inputs[0, colums];
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, colums];
                    if (item < min)
                    {
                        min = item;
                    }
                    if (item > max)
                    {
                        max = item;
                    }
                }
                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, colums] = (inputs[row, colums] - min) / (max - min);
                }
            }
            return result;
        }

        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);

                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                //стандартное квадратическое отклонение нейрона 
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standartError;
                }
            }
            return result;
        }

        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;
            var difference = actual - expected;
            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }
            for (int j = Layers.Count - 2; j>=0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];
                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];
                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error,Topology.LearningRate);
                    }
                }
            }
            var result = difference * difference;
            return result;
        }
    }
}
