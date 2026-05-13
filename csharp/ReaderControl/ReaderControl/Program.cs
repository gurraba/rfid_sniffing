using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection.PortableExecutable;
using System.Timers;
using System.Transactions;
using Impinj.OctaneSdk;
using RfidReaderCapture;

namespace RfidReaderCapture
{
    // ========================================================================
    // Class for reader (has state - connection, event handlers)
    // ========================================================================

    class ReaderCapture
    {
        private ImpinjReader reader;
        private List<TagRead> capturedTags;
        private bool isCapturing;
        private int targetReadCount;
        private int currentReadCount;
    
        
  
        public ReaderCapture(string hostname)
        {
            reader = new ImpinjReader();
            reader.Connect(hostname);
            capturedTags = new List<TagRead>();
            isCapturing = false;

            Console.WriteLine($"[Reader] Connected to {hostname}");
        }
        
        
        public void Configure(int antennaPort = 1, int session = 0)
        {
            Console.WriteLine("[Reader] Configuring...");

            Settings settings = reader.QueryDefaultSettings();

            // Report settings
            settings.Report.IncludeAntennaPortNumber = true;
            settings.Report.IncludePeakRssi = true;
            settings.Report.IncludePhaseAngle = true;
            settings.Report.IncludeFirstSeenTime = true;
            settings.Report.Mode = ReportMode.Individual;  // Get each read separately

            // Antenna settings
            settings.Antennas.DisableAll();
            settings.Antennas.GetAntenna((ushort)antennaPort).IsEnabled = true;
            settings.Antennas.GetAntenna((ushort)antennaPort).MaxTxPower = true
                ;
            settings.Antennas.GetAntenna((ushort)antennaPort).MaxRxSensitivity = true;

            // RF Mode
            // settings.ReaderMode = ReaderMode.AutoSetDenseReader;
            settings.RfMode = 0; // Mode 0 uses fm0 encoding only
            
            settings.Session = (ushort)session;
            

            reader.ApplySettings(settings);
            Console.WriteLine("[Reader] Configured");
        }
        public void StartReading()
        {
            reader.TagsReported += OnTagsReported;
            reader.Start();
        }

        public void StopReading()
        {
            reader.Stop();
            reader.TagsReported -= OnTagsReported;
        }


        public void SetTagFilter(string targetEpc)
        {
            Settings settings = reader.QuerySettings();

            if (targetEpc == "ALL")
            {
                Console.WriteLine($"[Reader] Removing filter - reading all tags");
                settings.Filters.Mode = TagFilterMode.None;
            }
            else
            {
                Console.WriteLine($"[Reader] Setting filter for tag: {targetEpc}");
                settings.Filters.TagFilter1.MemoryBank = MemoryBank.Epc;
                settings.Filters.TagFilter1.BitPointer = BitPointers.Epc;
                settings.Filters.TagFilter1.TagMask = targetEpc;
                settings.Filters.Mode = TagFilterMode.OnlyFilter1;
            }

            reader.ApplySettings(settings);
        }

        //single 

        public List<TagRead> CaptureTag(string epc, int readCount = 1, double timeoutSeconds = 2)
        {
            capturedTags.Clear();
            isCapturing = true;
            targetReadCount = readCount;
            currentReadCount = 0;

            SetTagFilter(epc);

            // Start reader
            reader.TagsReported += OnTagsReported;
            reader.Start();

            var startTime = DateTime.UtcNow;
            var timeout = TimeSpan.FromSeconds(timeoutSeconds);

            while (currentReadCount < targetReadCount && DateTime.UtcNow - startTime < timeout)
            {
                System.Threading.Thread.Sleep(10);
            }

            if (currentReadCount < targetReadCount)
            {
                Console.WriteLine($"Reader timeout. Only got {currentReadCount}/{targetReadCount} reads");
            }

            // Stop reader
            reader.Stop();
            reader.TagsReported -= OnTagsReported;
            isCapturing = false;

            Console.WriteLine($"[Reader] Captured {capturedTags.Count} tag reads");

            return new List<TagRead>(capturedTags);
        }
        public TagRead ReadTagOnce(string epc, double timeoutSeconds = 2.0)
        {
            capturedTags.Clear();
            isCapturing = true;
            targetReadCount = 1;
            currentReadCount = 0;

            // Stop just long enough to apply filter, then restart
            reader.Stop();
            SetTagFilter(epc);
            reader.TagsReported -= OnTagsReported; // avoid double subscription
            reader.TagsReported += OnTagsReported;
            reader.Start();

            var startTime = DateTime.UtcNow;
            while (currentReadCount < 1 && DateTime.UtcNow - startTime < TimeSpan.FromSeconds(timeoutSeconds))
            {
                System.Threading.Thread.Sleep(10);
            }

            reader.Stop();
            reader.TagsReported -= OnTagsReported;
            isCapturing = false;

            if (capturedTags.Count == 0)
                throw new Exception($"Tag {epc} not detected within {timeoutSeconds}s");

            return capturedTags[0];
        }

        public List<TagRead> CaptureDuration(string epc, double durationSeconds)
        {
            capturedTags.Clear();
            isCapturing = true;
            targetReadCount = int.MaxValue; // never stop early on count
            currentReadCount = 0;

            SetTagFilter(epc);
            reader.TagsReported += OnTagsReported;
            reader.Start();

            System.Threading.Thread.Sleep((int)(durationSeconds * 1000));

            reader.Stop();
            reader.TagsReported -= OnTagsReported;
            isCapturing = false;

            Console.WriteLine($"[Reader] Captured {capturedTags.Count} reads in {durationSeconds}s");
            return new List<TagRead>(capturedTags);
        }


        private void OnTagsReported(ImpinjReader sender, TagReport report)
        {
            if (!isCapturing) return;

            foreach (Tag tag in report.Tags)
            {
                // Stop if we've hit target count
                if (currentReadCount >= targetReadCount)
                {
                    return;
                }

                var tagRead = new TagRead
                {
                    Timestamp = DateTime.UtcNow, // Use UTC for consistency
                    Epc = tag.Epc.ToString(),
                    AntennaPort = tag.AntennaPortNumber,
                    Phase = tag.PhaseAngleInRadians,
                    Rssi = tag.PeakRssiInDbm
                }; 

                capturedTags.Add(tagRead);
                currentReadCount++;
            }
        }



        public void Disconnect()
        {
            if (reader.IsConnected)
            {
                reader.Disconnect();
                Console.WriteLine("[Reader] Disconnected");
            }
        }
    }

    // ========================================================================
    // Data structure
    // ========================================================================

    class TagRead
    {
        public DateTime Timestamp { get; set; }
        public string Epc { get; set; }
        public ushort AntennaPort { get; set; }
        public double Phase { get; set; }
        public double Rssi { get; set; }
    }

    // ========================================================================
    // Functions for file I/O
    // ========================================================================

    static class FileUtils
    {
        public static void SaveToCsv(string filepath, List<TagRead> tags)
        {
            Console.WriteLine($"[Save] Writing to {filepath}...");

            using (StreamWriter writer = new StreamWriter(filepath))
            {
                // Header
                writer.WriteLine("Timestamp,EPC,Antenna,Phase,RSSI");

                // Data
                foreach (var tag in tags)
                {
                    // Use Unix timestamp for easier matching with Python
                    double unixTime = ((DateTimeOffset)tag.Timestamp).ToUnixTimeMilliseconds() / 1000.0;

                    writer.WriteLine($"{unixTime:F3},{tag.Epc},{tag.AntennaPort},{tag.Phase:F6},{tag.Rssi:F2}");
                }
            }

            Console.WriteLine($"[Save] Saved {tags.Count} reads");
        }

        public static void SaveMetadata(string filepath, string targetEpc,
                                       double duration, DateTime startTime)
        {
            using (StreamWriter writer = new StreamWriter(filepath))
            {
                writer.WriteLine("{");
                writer.WriteLine($"  \"target_epc\": \"{targetEpc}\",");
                writer.WriteLine($"  \"duration\": {duration},");
                writer.WriteLine($"  \"start_time\": \"{startTime:O}\"");
                writer.WriteLine("}");
            }

            Console.WriteLine($"[Save] Saved metadata: {filepath}");
        }



    }

    


    static class TagRegistry
    {
        public static readonly Dictionary<string, string> Tags = new Dictionary<string, string>
    {
            //{ "A", "E2801191A5030066F8E4A83C" },
            //{ "B", "E2801190A5030069454824D7"},
            //{"C", "E2801191A5030066F8E5529C"},
            //{"M", "E2801190A50300650000" },
            //{"N", "E2801191A5030066F8E5A3CC"}


            {"A", "E2801190A503006500001111" },
            {"B", "E2801190A5030063658F657D" },
            {"C", "E2801190A5030063658F658D" },
            {"D", "E2801190A50300650000"},
            {"E", "E2801190A503006500002222"},
            {"S", "E2801191A5030066F8E5"}


    };

        public static void PrintTags()
        {
            Console.WriteLine("\nAvailable tags:");
            foreach (var tag in Tags)
            {
                Console.WriteLine($"  {tag.Key}: {tag.Value}");
            }
        }
    }

    // ========================================================================
    // Main program
    // ========================================================================



    class ExperimentRunner
    {
        ReaderCapture reader;

        public ExperimentRunner( ReaderCapture reader, string readerIp) {
            this.reader = reader;
        }

        

        public void QuickTest(string epc)
        {
            Console.WriteLine("\nReading tag 5 with 100ms interval...\n");

            var results = new List<TagRead>();
            var starttime = DateTime.Now;

            for (int i = 0; i < 5; i++)
            {
                var read = reader.ReadTagOnce(epc);
                results.Add(read);
                Console.WriteLine($"Read {i + 1}: Phase={read.Phase:F3} rad, RSSI={read.Rssi:F1} dB");
                //System.Threading.Thread.Sleep(100);
            }

            
                SaveResults("quick_test", results, epc, starttime, duration: 0.5);
        }

        public void ReadInSequence(List<string> epcs, int readsPerTag)
        {

            //reader.StartReading();


            Console.WriteLine($"\nReading {epcs.Count} tags, {readsPerTag} reads each\n");

            var results = new List<TagRead>();
            var starttime = DateTime.Now;


            for (int i = 0; i < readsPerTag; i++)
            {
                foreach (var epc in epcs)
                {
                    Console.WriteLine($"Reading tag {epc}...");
                    var read = reader.ReadTagOnce(epc);
                    results.Add(read);
                    Console.WriteLine($"  {i + 1}/{readsPerTag}: Phase={read.Phase:F3}, Time={read.Timestamp:HH:mm:ss.fff}");
                }
            }

            var duration = (DateTime.UtcNow - starttime).TotalSeconds;
            var epclist = string.Join(",", epcs);

            //reader.StopReading();

            SaveResults("sequence", results, epclist, starttime, duration);
           
        }

        public void ReadAnyTags(double durationSeconds)
        {
            Console.WriteLine($"\nReading all tags for {durationSeconds}s...\n");

            var startTime = DateTime.UtcNow;

            // Use "ALL" to disable filter
            var results = reader.CaptureTag("ALL", int.MaxValue, durationSeconds);

            Console.WriteLine($"\nFound {results.Count} tag reads from {results.Select(r => r.Epc).Distinct().Count()} unique tags");

            // Print unique EPCs found
            var uniqueEpcs = results.Select(r => r.Epc).Distinct();
            foreach (var epc in uniqueEpcs)
            {
                var count = results.Count(r => r.Epc == epc);
                Console.WriteLine($"  {epc}: {count} reads");
            }
            
            var duration = (DateTime.UtcNow - startTime).TotalSeconds;
            SaveResults("any_tags", results, "ALL", startTime, duration);
        }

            public void readDuringPeriod(string epc, double durationSeconds)
            {
            Console.WriteLine($"\nReading tag {TagRegistry.Tags.GetValueOrDefault(epc)} for {durationSeconds}s...\n");
            var startTime = DateTime.UtcNow;
            var results = reader.CaptureDuration(epc, durationSeconds);
            SaveResults("continous", results, epc, startTime, durationSeconds);


            }

        public static void SaveResults(string prefix, List<TagRead> results, string targetEPC, DateTime starttime, double duration)
        {
            Console.Write("Name your experiment: "); 
            var input =  Console.ReadLine();
            prefix = string.IsNullOrEmpty(input) ? prefix : input;

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var filename = $"{prefix}_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
            filename = prefix;
            var csvpath = $"C:/Users/gusta/Documents/programmering/RFID_project/data/captures/{filename}_reader.csv";
            var metapath = $"C:/Users/gusta/Documents/programmering/RFID_project/data/captures/{filename}_reader_meta.json";

            FileUtils.SaveToCsv(csvpath, results);
            FileUtils.SaveMetadata(metapath, targetEPC, duration, starttime);
            Console.WriteLine($"\nSaved data and metadata: {filename}");
        }



    }
}






class Program
{
    static void Main(string[] args)
    {

        // Parse arguments
        string readerIp = "169.254.1.1";
        string readerIp2 = "169.254.72.166";

        try
        {


            // Connect and configure

            // List of potential IP addresses to try
            string[] readerIps = { "169.254.72.166", "169.254.1.1" };
            ReaderCapture reader = null;
            ExperimentRunner experients = null;
            bool isConnected = false;

            Console.WriteLine("=================================================");
            Console.WriteLine("RFID Reader Capture Program");

            foreach (string ip in readerIps)
            {
                try
                {
                    Console.WriteLine($"\n===================== CONNECTING TO {ip} ====================");

                    // Attempt to initialize
                    var tempReader = new ReaderCapture(ip);

                    // Some libraries don't throw until you actually try to Configure or Connect
                    tempReader.Configure(antennaPort: 1, session: 0);

                    // If we reach this line, the connection was successful
                    reader = tempReader;
                    experients = new ExperimentRunner(reader, ip);
                    isConnected = true;

                    Console.WriteLine($"Successfully connected to {ip}!");
                    break; // Exit the loop since we found a working IP
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to connect to {ip}: {ex.Message}");
                    // Loop continues to the next IP address
                }
            }

            if (!isConnected)
            {
                Console.WriteLine("\n[ERROR] Could not connect to any of the provided IP addresses.");
                return; // Or handle the failure appropriately
            }





            while (true)
            {
                Console.WriteLine("\n" + new string('=', 50));
                Console.WriteLine("RFID Experiment");
                Console.WriteLine(new string('=', 50));
                Console.WriteLine("\n1. Quick test (1 tag, 5 reads)");
                Console.WriteLine("2. Read tags in sequence");
                Console.WriteLine("3. Read any tags found");
                Console.WriteLine("4, Read during period");
                Console.WriteLine("5. Exit");

                Console.WriteLine("\nChoice: ");
                var choice = Console.ReadLine();

                if (choice == "1")
                {
                    TagRegistry.PrintTags();
                    Console.WriteLine("Select tag: ");
                    var tagId = Console.ReadLine().ToUpper();

                    if (TagRegistry.Tags.ContainsKey(tagId))
                    {
                        //System.Threading.Thread.Sleep(10000);
                        experients.QuickTest(TagRegistry.Tags[tagId]);
                    }
                    else
                    {
                        Console.WriteLine("Invalid tag EPC");
                    }
                }
                else if (choice == "2")
                {
                    TagRegistry.PrintTags();
                    Console.Write("Select tags (e.g ABC): ");
                    var input = Console.ReadLine().ToUpper();
                    var epcs = new List<string>();
                    foreach (var c in input)
                    {
                        var key = c.ToString();
                        if (TagRegistry.Tags.ContainsKey(key))
                        {
                            epcs.Add(TagRegistry.Tags[key]);
                        }
                    }

                    if (epcs.Count == 0)
                    {
                        Console.WriteLine("No valid tags selected");
                        continue;
                    }

                    Console.Write("Reads per tag [5]: ");
                    var readsStr = Console.ReadLine();
                    int reads = string.IsNullOrEmpty(readsStr) ? 5 : int.Parse(readsStr);
                    //System.Threading.Thread.Sleep(10000);
                    experients.ReadInSequence(epcs, reads);
                }
                else if(choice == "3")
                {
                    Console.Write("For how many seconds?:");
                    int input = Convert.ToInt32(Console.ReadLine());

                    experients.ReadAnyTags(input);
                }
                else if(choice == "4")
                {
                    Console.WriteLine("Which tag? (e.g ABC): ");
                    var key = Console.ReadLine().ToUpper();
                    string epc = "ALL";
                    if (TagRegistry.Tags.ContainsKey(key))
                    {
                        epc = TagRegistry.Tags[key];
                    }
                    else
                    {
                        Console.WriteLine("Invalid Tag selection");
                    }

                    Console.WriteLine("For how many seconds? [5] : ");
                    var readsStr = Console.ReadLine();
                    int duration = string.IsNullOrEmpty(readsStr) ? 5 : int.Parse(readsStr);
                    experients.readDuringPeriod(epc, duration);
                }

                else if (choice == "5")
                {
                    Console.WriteLine("Disconnecting reader");
                    reader.Disconnect();
                    Console.WriteLine("Goodbye!");
                    break;
                }
                else
                {
                    Console.WriteLine("Invalid choice.");
                }
            }

        }


        catch (OctaneSdkException e)
        {
            Console.WriteLine($"Octane SDK exception: {e.Message}");
        }
        catch (Exception e)
        {
            Console.WriteLine($"Exception: {e.Message}");
        }

    }
}