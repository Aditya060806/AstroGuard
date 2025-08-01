import { motion } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Rocket, RotateCcw, Database, Zap, TrendingUp, CheckCircle } from 'lucide-react';

const syncSteps = [
  {
    id: '1',
    title: 'Object Evolution',
    description: 'Real-world space station objects tracked and catalogued in high-fidelity detail',
    icon: Database,
    status: 'completed',
    color: 'text-primary',
    details: '12,000+ objects scanned with millimeter precision'
  },
  {
    id: '2',
    title: 'Falcon Synthesis',
    description: 'Digital twin environment generates thousands of training variations',
    icon: Rocket,
    status: 'completed',
    color: 'text-accent',
    details: '50K synthetic images generated with physics simulation'
  },
  {
    id: '3',
    title: 'Model Retraining',
    description: 'AI model updated with enhanced synthetic dataset for improved accuracy',
    icon: RotateCcw,
    status: 'active',
    color: 'text-primary',
    details: 'Training epoch 87/100 - ETA 2.3 hours'
  },
  {
    id: '4',
    title: 'Deployment',
    description: 'Updated model deployed to orbital detection systems with improved metrics',
    icon: TrendingUp,
    status: 'pending',
    color: 'text-muted-foreground',
    details: 'Scheduled for next maintenance window'
  },
];

const improvements = [
  { metric: 'mAP@0.5', before: '91.2%', after: '94.7%', improvement: '+3.5%' },
  { metric: 'Precision', before: '96.1%', after: '99.2%', improvement: '+3.1%' },
  { metric: 'Recall', before: '89.7%', after: '91.4%', improvement: '+1.7%' },
  { metric: 'False Positives', before: '8.3%', after: '4.2%', improvement: '-4.1%' },
];

export default function FalconSync() {
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
          <div className="flex flex-col sm:flex-row items-center justify-center mb-4">
            <Rocket className="w-6 h-6 sm:w-8 sm:h-8 text-primary mb-2 sm:mb-0 sm:mr-3" />
            <h2 className="text-3xl sm:text-4xl font-bold">Falcon Model Sync</h2>
          </div>
          <p className="text-lg sm:text-xl text-muted-foreground">
            Continuous model improvement through synthetic data generation
          </p>
        </motion.div>

        {/* Sync Timeline */}
        <motion.div
          className="relative mb-12 sm:mb-16 px-4"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
        >
          {/* Timeline Line - Hidden on mobile, shown on desktop */}
          <div className="hidden lg:block absolute left-1/2 transform -translate-x-1/2 w-1 h-full bg-gradient-to-b from-primary via-accent to-muted opacity-30" />

          <div className="space-y-8 sm:space-y-12">
            {syncSteps.map((step, index) => (
              <motion.div
                key={step.id}
                className={`flex items-center ${index % 2 === 0 ? 'lg:flex-row' : 'lg:flex-row-reverse'} flex-col lg:flex-row`}
                initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                viewport={{ once: true }}
              >
                <div className={`w-full lg:w-1/2 ${index % 2 === 0 ? 'lg:pr-8 lg:text-right' : 'lg:pl-8'} mb-4 lg:mb-0`}>
                  <Card className="glass-panel hover-tilt hover:cosmic-glow transition-all duration-500 group cursor-pointer">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className={`p-3 rounded-lg bg-gradient-cosmic ${step.color}`}>
                          <step.icon className="w-6 h-6" />
                        </div>
                        <Badge
                          variant={
                            step.status === 'completed' ? 'default' :
                            step.status === 'active' ? 'secondary' : 'outline'
                          }
                          className="capitalize"
                        >
                          {step.status}
                        </Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <h3 className="text-lg font-semibold">{step.title}</h3>
                        <p className="text-sm text-muted-foreground">{step.description}</p>
                        <p className="text-xs text-primary font-medium">{step.details}</p>
                      </div>

                      {step.status === 'active' && (
                        <div className="pt-4">
                          <div className="w-full bg-secondary rounded-full h-2">
                            <motion.div
                              className="bg-gradient-to-r from-primary to-accent h-2 rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: '87%' }}
                              transition={{ duration: 2, ease: "easeOut" }}
                            />
                          </div>
                          <div className="flex justify-between text-xs text-muted-foreground mt-1">
                            <span>Progress</span>
                            <span>87%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </Card>
                </div>

                {/* Timeline Node - Hidden on mobile */}
                <div className="relative z-10 hidden lg:block">
                  <motion.div
                    className={`w-4 h-4 rounded-full border-4 ${
                      step.status === 'completed' ? 'bg-primary border-primary' :
                      step.status === 'active' ? 'bg-accent border-accent animate-pulse' :
                      'bg-muted border-muted'
                    }`}
                    whileHover={{ scale: 1.5 }}
                    transition={{ duration: 0.2 }}
                  />
                  {step.status === 'completed' && (
                    <motion.div
                      className="absolute inset-0 flex items-center justify-center"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.3, delay: index * 0.2 + 0.5 }}
                    >
                      <CheckCircle className="w-4 h-4 text-primary" />
                    </motion.div>
                  )}
                </div>

                <div className="hidden lg:block lg:w-1/2" />
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Performance Improvements */}
        <motion.div
          className="grid lg:grid-cols-2 gap-6 sm:gap-8 px-4"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
        >
          {/* Before/After Metrics */}
          <Card className="glass-panel hover-tilt hover:cosmic-glow">
            <h3 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Performance Improvements</h3>
            
            <div className="space-y-4">
              {improvements.map((improvement, index) => (
                <motion.div
                  key={improvement.metric}
                  className="flex items-center justify-between p-3 rounded-lg bg-card/50"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <span className="font-medium">{improvement.metric}</span>
                  
                  <div className="flex items-center space-x-4 text-sm">
                    <span className="text-muted-foreground">{improvement.before}</span>
                    <span className="text-muted-foreground">→</span>
                    <span className="text-primary font-semibold">{improvement.after}</span>
                    <Badge variant="default" className="text-xs">
                      {improvement.improvement}
                    </Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          </Card>

          {/* Sync Controls */}
          <Card className="glass-panel hover-tilt hover:aurora-glow">
            <h3 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6 bg-gradient-to-r from-accent to-primary bg-clip-text text-transparent">Sync Management</h3>
            
            <div className="space-y-6">
              {/* Current Status */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Current Status</span>
                  <Badge variant="secondary" className="animate-pulse">Training Active</Badge>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Last Sync</span>
                  <span className="text-sm">2024-07-28 14:32 UTC</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Next Scheduled</span>
                  <span className="text-sm">2024-08-04 02:00 UTC</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="space-y-3 pt-4 border-t border-border/50">
                <Button className="w-full cosmic-glow" disabled>
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Training in Progress...
                </Button>
                
                <Button variant="outline" className="w-full">
                  <Database className="w-4 h-4 mr-2" />
                  View Training Logs
                </Button>
                
                <Button variant="ghost" className="w-full">
                  <Zap className="w-4 h-4 mr-2" />
                  Schedule Manual Sync
                </Button>
              </div>

              {/* Next Update Preview */}
              <div className="glass rounded-lg p-4 bg-gradient-cosmic/20">
                <h4 className="font-medium mb-2 text-primary">Next Update Preview</h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Enhanced detection for low-light scenarios and improved recognition of partially occluded objects. 
                  Expected mAP improvement: +1.2%
                </p>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}