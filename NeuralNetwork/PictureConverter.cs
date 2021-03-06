﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NeuralNetwork
{
    public class PictureConverter
    {
        public int boundary { get; set; } = 128;
        public int Height { get; set; }
        public int Widht { get; set; }
        public List<int> Convert(string path)
        {
            var result = new List<int>();
            Bitmap image = new Bitmap(path);
            Height = image.Height;
            Widht = image.Width;
            
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    var value = Brightness(pixel);
                    result.Add(value);
                }
            }
            return result;
        }

        private int Brightness(Color pixel)
        {
            var result = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
            return result < boundary ? 0:1;
        }
        public void Save(string path,  List<int> pixels)
        {
            var image = new Bitmap(Widht, Height);
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0;x < image.Width; x++)
                {
                    var color = pixels[y * Widht + x] == 1 ? Color.White : Color.Black;
                    image.SetPixel(x, y, color);
                }
            }
            image.Save(path);
        }

    }

}
