import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, Zap, Target, Clock, Cpu, Database } from 'lucide-react';
import { useEffect, useState } from 'react';

const metrics = [
  { label: 'mAP@0.5', value: 94.7, suffix: '%', change: '+2.3%', trend: 'up', icon: Target, color: 'text-primary' },
  { label: 'Precision', value: 99.2, suffix: '%', change: '+0.8%', trend: 'up', icon: Zap, color: 'text-accent' },
  { label: 'Recall', value: 91.4, suffix: '%', change: '+1.2%', trend: 'up', icon: TrendingUp, color: 'text-primary' },
  { label: 'Inference Speed', value: 23, suffix: 'ms', change: '-3ms', trend: 'up', icon: Clock, color: 'text-accent' },
  { label: 'Model Size', value: 47.2, suffix: 'MB', change: '-2.1MB', trend: 'up', icon: Cpu, color: 'text-primary' },
  { label: 'Training Samples', value: 12.4, suffix: 'K', change: '+2.1K', trend: 'up', icon: Database, color: 'text-accent' },
];

const confusionMatrix = [
  [92, 3, 1],
  [2, 89, 4],
  [1, 5, 87],
];

const classNames = ['Fire Ext.', 'Toolbox', 'O2 Tank'];

// Animated counter hook
const useAnimatedCounter = (end: number, start: number = 0, duration: number = 2000) => {
  const [count, setCount] = useState(start);
  const [hasAnimated, setHasAnimated] = useState(false);

  useEffect(() => {
    if (!hasAnimated) return;
    
    let startTime: number;
    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = (currentTime - startTime) / duration;
      
      if (progress < 1) {
        setCount(Math.floor(start + (end - start) * progress));
        requestAnimationFrame(animate);
      } else {
        setCount(end);
      }
    };
    
    requestAnimationFrame(animate);
  }, [end, start, duration, hasAnimated]);

  return { count, setHasAnimated };
};

export default function MetricsDashboard() {
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
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">Mission Metrics</h2>
          <p className="text-lg sm:text-xl text-muted-foreground">Real-time performance analytics from orbital deployment</p>
        </motion.div>

        {/* Key Metrics Grid */}
        <motion.div
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 mb-8 sm:mb-12 px-4"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          viewport={{ once: true }}
        >
        {metrics.map((metric, index) => {
            const { count, setHasAnimated } = useAnimatedCounter(metric.value, 0, 2000);
            
            return (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ 
                  opacity: 1, 
                  scale: 1,
                  transition: { 
                    duration: 0.5, 
                    delay: index * 0.1,
                    onComplete: () => setHasAnimated(true)
                  }
                }}
                viewport={{ once: true }}
              >
                <Card className="glass-panel hover-tilt hover:orbital-glow transition-all duration-500 cursor-pointer group">
                  <div className="flex items-center justify-between mb-4">
                    <motion.div 
                      className={`p-3 rounded-xl bg-gradient-cosmic ${metric.color} relative`}
                      whileHover={{ scale: 1.1, rotate: 5 }}
                      transition={{ duration: 0.3 }}
                    >
                      <metric.icon className="w-6 h-6" />
                      <div className="absolute inset-0 bg-gradient-aurora opacity-0 group-hover:opacity-20 rounded-xl transition-opacity duration-300" />
                    </motion.div>
                    <Badge 
                      variant={metric.trend === 'up' ? 'default' : 'secondary'} 
                      className="text-xs cosmic-glow"
                    >
                      {metric.change}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground font-medium">{metric.label}</p>
                    <motion.p 
                      className="text-3xl font-bold counter group-hover:text-primary transition-colors duration-300"
                      initial={{ scale: 0.8 }}
                      whileInView={{ scale: 1 }}
                      transition={{ duration: 0.3, delay: index * 0.1 + 0.5 }}
                    >
                      {count.toFixed(metric.suffix === 'ms' || metric.suffix === 'MB' || metric.suffix === 'K' ? 1 : 1)}{metric.suffix}
                    </motion.p>
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-6 sm:gap-8 px-4">
          {/* Training Progress */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            viewport={{ once: true }}
          >
            <Card className="glass-panel hover-tilt hover:cosmic-glow">
              <h3 className="text-2xl font-semibold mb-6 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Training Progress</h3>
              
              <div className="space-y-6">
                {/* Loss Chart Visualization */}
                <div className="h-48 relative bg-gradient-to-br from-secondary/20 to-card/50 rounded-lg p-4">
                  <div className="absolute inset-4">
                    {/* Simulated training loss curve */}
                    <svg className="w-full h-full" viewBox="0 0 300 120">
                      <defs>
                        <linearGradient id="lossGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="hsl(var(--destructive))" />
                          <stop offset="100%" stopColor="hsl(var(--primary))" />
                        </linearGradient>
                      </defs>
                      <path
                        d="M 0 100 Q 50 80 100 60 Q 150 45 200 35 Q 250 30 300 25"
                        stroke="url(#lossGradient)"
                        strokeWidth="3"
                        fill="none"
                        className="animate-pulse-glow"
                      />
                      {/* Data points */}
                      {[0, 50, 100, 150, 200, 250, 300].map((x, i) => {
                        const y = 100 - (i * 12) - (i > 2 ? 5 : 0);
                        return (
                          <circle
                            key={i}
                            cx={x}
                            cy={y}
                            r="3"
                            fill="hsl(var(--primary))"
                            className="animate-pulse"
                            style={{ animationDelay: `${i * 0.2}s` }}
                          />
                        );
                      })}
                    </svg>
                  </div>
                  
                  <div className="absolute bottom-2 left-4 text-xs text-muted-foreground">
                    Epoch 0
                  </div>
                  <div className="absolute bottom-2 right-4 text-xs text-muted-foreground">
                    Epoch 100
                  </div>
                  <div className="absolute top-2 left-4 text-xs text-muted-foreground">
                    Loss: 0.023
                  </div>
                </div>

                {/* Training Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">Total Epochs</p>
                    <p className="text-xl font-bold text-primary">156</p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">Training Time</p>
                    <p className="text-xl font-bold text-accent">14.2h</p>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Confusion Matrix */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            viewport={{ once: true }}
          >
            <Card className="glass-panel hover-tilt hover:aurora-glow">
              <h3 className="text-2xl font-semibold mb-6 bg-gradient-to-r from-accent to-primary bg-clip-text text-transparent">Confusion Matrix</h3>
              
              <div className="space-y-4">
                {/* Matrix Grid */}
                <div className="grid grid-cols-4 gap-2 text-center">
                  {/* Headers */}
                  <div className="text-xs text-muted-foreground"></div>
                  {classNames.map((className) => (
                    <div key={className} className="text-xs text-muted-foreground p-2">
                      {className}
                    </div>
                  ))}
                  
                  {/* Matrix Rows */}
                  {confusionMatrix.map((row, rowIndex) => (
                    <div key={rowIndex} className="contents">
                      <div className="text-xs text-muted-foreground p-2 text-right">
                        {classNames[rowIndex]}
                      </div>
                      {row.map((value, colIndex) => {
                        const isCorrect = rowIndex === colIndex;
                        const intensity = value / 100;
                        return (
                          <motion.div
                            key={colIndex}
                            className={`aspect-square rounded-lg flex items-center justify-center text-sm font-medium transition-all duration-300 hover:scale-110 cursor-pointer ${
                              isCorrect 
                                ? 'bg-primary/20 text-primary border border-primary/30' 
                                : 'bg-destructive/20 text-destructive border border-destructive/30'
                            }`}
                            style={{
                              backgroundColor: isCorrect 
                                ? `hsl(var(--primary) / ${0.1 + intensity * 0.3})` 
                                : `hsl(var(--destructive) / ${0.1 + (1 - intensity) * 0.2})`
                            }}
                            initial={{ opacity: 0, scale: 0.8 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.3, delay: (rowIndex * 3 + colIndex) * 0.1 }}
                            viewport={{ once: true }}
                          >
                            {value}
                          </motion.div>
                        );
                      })}
                    </div>
                  ))}
                </div>

                {/* Legend */}
                <div className="flex items-center justify-between text-xs text-muted-foreground pt-4 border-t border-border/50">
                  <span>Predicted</span>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <div className="w-3 h-3 bg-primary/30 rounded"></div>
                      <span>Correct</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <div className="w-3 h-3 bg-destructive/30 rounded"></div>
                      <span>Incorrect</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </section>
  );
}