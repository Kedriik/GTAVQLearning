using System;
using System.Windows.Forms;
using GTA;
using GTA.Math;
using System.Diagnostics;
using System.IO.Pipes;
using System.IO;
using System.Threading;
namespace GTATimer
{
    public class GTATimerWithSpeedReward : Script
    {
        static Stopwatch stopWatch = new Stopwatch();
        private static Mutex mut = new Mutex();
        static TimeSpan ts;
        static bool stopwatchStarted = false;
        static string elapsedTime;
        static NamedPipeServerStream server;
        static float V = -11.1f;
        public GTATimerWithSpeedReward(){
            Tick += onTick;
            KeyUp += onKeyUp;
            Thread t = new Thread(new ThreadStart(provideV));
            server = new NamedPipeServerStream("VStreaming");
            server.WaitForConnection();
            t.Start();
        }
        public static void provideV(){
            
            
            while (true) {
                try {
                    mut.WaitOne();
                    var byteArray = new byte[4];
                    byteArray = BitConverter.GetBytes(GTATimerWithSpeedReward.V);
                    server.Write(byteArray, 0, 4);
                    
                    mut.ReleaseMutex();
                }catch(Exception e){
                    //server.WaitForConnection();
                    break;
                }
                Thread.Sleep(1);
            }
            server.Disconnect();
            server.Close();
        }
        protected static void onTick(object sender, EventArgs e)
        {
            ts = stopWatch.Elapsed;
            elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);;
            UI.ShowSubtitle("Runtime " + elapsedTime);
            mut.WaitOne();
            if (Game.Player.Character.CurrentVehicle.HasCollidedWithAnything == false)
            {
                GTATimerWithSpeedReward.V = Game.Player.Character.CurrentVehicle.Speed;
            }
            else
            {
                GTATimerWithSpeedReward.V = -1000;
                resetPosition();
            }
            mut.ReleaseMutex();
        }
        protected static void onKeyUp(object value0, System.Windows.Forms.KeyEventArgs value1)
        {
            if(stopwatchStarted == false && value1.KeyCode == Keys.T)
            {
                resetPosition();
                stopWatch.Reset();
                stopWatch.Start();
                stopwatchStarted = true;
            }
            else if (stopwatchStarted == true && value1.KeyCode == Keys.T)
            {
                ts = stopWatch.Elapsed;
                stopWatch.Stop();
                stopwatchStarted = false;
                elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            }
        }
        static void resetPosition()
        {
            Game.Player.Character.LastVehicle.Position = new Vector3(169.1f, 6561.4f, 31.3f);
            Game.Player.Character.CurrentVehicle.Rotation = new Vector3(0.52f, 0.61f, -146.0f);
            Game.Player.Character.CurrentVehicle.Speed = 0;
            Game.Player.Character.CurrentVehicle.Repair();
        }
    }
}
