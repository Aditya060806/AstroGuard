import { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Upload, Play, Pause, RotateCcw, Filter, Eye, EyeOff } from 'lucide-react';

interface DetectedObject {
  id: string;
  label: string;
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  color: string;
}

const mockDetections: DetectedObject[] = [
  { id: '1', label: 'Fire Extinguisher', confidence: 0.94, bbox: { x: 120, y: 80, width: 60, height: 120 }, color: '#ef4444' },
  { id: '2', label: 'Toolbox', confidence: 0.87, bbox: { x: 250, y: 150, width: 80, height: 50 }, color: '#3b82f6' },
  { id: '3', label: 'Oxygen Tank', confidence: 0.92, bbox: { x: 50, y: 200, width: 40, height: 80 }, color: '#10b981' },
];

export default function DetectionLab() {
  const [confidenceThreshold, setConfidenceThreshold] = useState([0.8]);
  const [showLabels, setShowLabels] = useState(true);
  const [selectedFilters, setSelectedFilters] = useState<string[]>(['all']);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const filteredDetections = mockDetections.filter(detection => 
    detection.confidence >= confidenceThreshold[0] && 
    (selectedFilters.includes('all') || selectedFilters.includes(detection.label.toLowerCase().replace(' ', '')))
  );

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setIsUploading(true);
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        setIsUploading(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleProcessImage = async () => {
    setIsProcessing(true);
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsProcessing(false);
  };

  return (
    <section className="py-20 px-4 relative">
      <div className="container mx-auto">
        <motion.div
          className="text-center mb-8 sm:mb-12 px-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">Detection Laboratory</h2>
          <p className="text-lg sm:text-xl text-muted-foreground">Upload space station images and see our AI in action</p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6 sm:gap-8 px-4">
          {/* Controls Panel */}
          <motion.div
            className="space-y-4 sm:space-y-6 order-2 lg:order-1"
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
          >
            <Card className="glass-panel hover-tilt space-y-4 sm:space-y-6">
              <h3 className="text-base sm:text-lg font-semibold flex items-center">
                <Filter className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
                Detection Controls
              </h3>

              {/* Upload Button */}
              <div className="relative">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  id="image-upload"
                />
                <Button 
                  className="w-full" 
                  variant="outline"
                  disabled={isUploading}
                >
                  {isUploading ? (
                    <>
                      <div className="w-4 h-4 mr-2 border-2 border-current border-t-transparent rounded-full animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Image
                    </>
                  )}
                </Button>
              </div>

              {/* Confidence Threshold */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Confidence Threshold: {confidenceThreshold[0]}</label>
                <Slider
                  value={confidenceThreshold}
                  onValueChange={setConfidenceThreshold}
                  max={1}
                  min={0.1}
                  step={0.05}
                  className="w-full"
                />
              </div>

              {/* Filter Classes */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Filter Classes</label>
                <div className="flex flex-wrap gap-2">
                  {['all', 'fireextinguisher', 'toolbox', 'oxygentank'].map((filter) => (
                    <Badge
                      key={filter}
                      variant={selectedFilters.includes(filter) ? "default" : "outline"}
                      className="cursor-pointer capitalize text-xs hover:scale-105 transition-transform"
                      onClick={() => {
                        if (filter === 'all') {
                          setSelectedFilters(['all']);
                        } else {
                          const newFilters = selectedFilters.includes(filter)
                            ? selectedFilters.filter(f => f !== filter)
                            : [...selectedFilters.filter(f => f !== 'all'), filter];
                          setSelectedFilters(newFilters.length > 0 ? newFilters : ['all']);
                        }
                      }}
                    >
                      {filter === 'fireextinguisher' ? 'Fire Extinguisher' : 
                       filter === 'oxygentank' ? 'Oxygen Tank' : 
                       filter.charAt(0).toUpperCase() + filter.slice(1)}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Toggle Labels */}
              <Button
                variant="ghost"
                className="w-full justify-start"
                onClick={() => setShowLabels(!showLabels)}
              >
                {showLabels ? <Eye className="w-4 h-4 mr-2" /> : <EyeOff className="w-4 h-4 mr-2" />}
                {showLabels ? 'Hide Labels' : 'Show Labels'}
              </Button>

              {/* Process Button */}
              <Button 
                className="w-full cosmic-glow" 
                onClick={handleProcessImage}
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <>
                    <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Run Detection
                  </>
                )}
              </Button>
            </Card>

            {/* Detection Results */}
            <Card className="glass-panel hover-tilt space-y-4">
              <h3 className="text-lg font-semibold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Detection Results</h3>
              <div className="space-y-3">
                {filteredDetections.map((detection) => (
                  <motion.div
                    key={detection.id}
                    className="flex items-center justify-between p-3 rounded-lg bg-card/50"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="flex items-center space-x-3">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: detection.color }}
                      />
                      <span className="font-medium">{detection.label}</span>
                    </div>
                    <Badge variant="secondary">
                      {(detection.confidence * 100).toFixed(1)}%
                    </Badge>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>

          {/* Main Detection View */}
          <motion.div
            className="lg:col-span-2 order-1 lg:order-2"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            viewport={{ once: true }}
          >
            <Card className="glass-panel hover-tilt orbital-glow overflow-hidden">
              <div className="relative aspect-video sm:aspect-[16/10] lg:aspect-video bg-gradient-to-br from-card to-secondary/20">
                {/* Uploaded Image or Mock Background */}
                {uploadedImage ? (
                  <img 
                    src={uploadedImage} 
                    alt="Uploaded space station" 
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                ) : (
                  <div className="absolute inset-0 bg-gradient-to-br from-secondary/30 to-card/50" />
                )}
                
                {/* Detection Bounding Boxes */}
                {filteredDetections.map((detection) => (
                  <motion.div
                    key={detection.id}
                    className="absolute pointer-events-none"
                    style={{
                      left: `${detection.bbox.x / 400 * 100}%`,
                      top: `${detection.bbox.y / 300 * 100}%`,
                      width: `${detection.bbox.width / 400 * 100}%`,
                      height: `${detection.bbox.height / 300 * 100}%`,
                    }}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <div 
                      className="absolute inset-0 border-2 rounded-lg shadow-lg"
                      style={{ 
                        borderColor: detection.color,
                        boxShadow: `0 0 20px ${detection.color}30`
                      }}
                    />
                    {showLabels && (
                      <motion.div
                        className="absolute -top-8 left-0 glass rounded-md px-2 py-1 text-xs whitespace-nowrap z-10"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, delay: 0.5 }}
                        style={{ backgroundColor: `${detection.color}20` }}
                      >
                        <span style={{ color: detection.color }} className="font-medium">
                          {detection.label}
                        </span>
                        <span className="ml-2 text-muted-foreground">
                          {(detection.confidence * 100).toFixed(1)}%
                        </span>
                      </motion.div>
                    )}
                  </motion.div>
                ))}

                {/* Processing Overlay */}
                {isProcessing && (
                  <motion.div
                    className="absolute inset-0 bg-background/80 flex items-center justify-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <div className="text-center space-y-4">
                      <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
                      <p className="text-lg font-medium">Analyzing orbital imagery...</p>
                    </div>
                  </motion.div>
                )}

                {/* Mock Space Objects */}
                <div className="absolute inset-0 overflow-hidden opacity-30">
                  <div className="absolute top-1/4 left-1/4 w-16 h-24 bg-destructive/60 rounded-lg transform rotate-12" />
                  <div className="absolute top-1/2 right-1/3 w-20 h-12 bg-primary/60 rounded-md" />
                  <div className="absolute bottom-1/3 left-1/6 w-10 h-20 bg-accent/60 rounded-full" />
                </div>
              </div>
              
              {/* Image Info */}
              <div className="p-3 sm:p-4 border-t border-border/50">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs sm:text-sm text-muted-foreground">
                  <span>ISS Module Alpha - EVA Camera 03</span>
                  <span>2024-07-30 15:42:33 UTC</span>
                </div>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </section>
  );
}