import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Satellite, Menu, X, Target, BarChart3, AlertTriangle, Rocket, Smartphone, FileText } from 'lucide-react';

const navItems = [
  { label: 'Detection Lab', href: '#detection', icon: Target },
  { label: 'Metrics', href: '#metrics', icon: BarChart3 },
  { label: 'Failures', href: '#failures', icon: AlertTriangle },
  { label: 'Falcon Sync', href: '#falcon', icon: Rocket },
  { label: 'Mobile', href: '#mobile', icon: Smartphone },
  { label: 'Reports', href: '#reports', icon: FileText },
];

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('');

  useEffect(() => {
    const handleScroll = () => {
      const sections = navItems.map(item => item.href.slice(1));
      const scrollPosition = window.scrollY + 100;

      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const offsetTop = element.offsetTop;
          const offsetBottom = offsetTop + element.offsetHeight;

          if (scrollPosition >= offsetTop && scrollPosition < offsetBottom) {
            setActiveSection(section);
            break;
          }
        }
      }

      // Update progress bar
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = window.scrollY / totalHeight;
      const progressBar = document.querySelector('.scroll-progress');
      if (progressBar) {
        (progressBar as HTMLElement).style.transform = `scaleX(${progress})`;
      }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial call
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (href: string) => {
    const element = document.getElementById(href.slice(1));
    if (element) {
      const offset = 80; // Account for fixed navbar
      const targetPosition = element.offsetTop - offset;
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
    setIsOpen(false);
  };

  return (
    <>
      {/* Main Navigation */}
      <motion.nav
        className="fixed top-0 left-0 right-0 z-50 glass border-b border-border/50"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <motion.button
              className="flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.2 }}
              onClick={() => scrollToSection('#')}
            >
              <div className="p-2 rounded-lg bg-gradient-cosmic">
                <Satellite className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="font-bold text-lg">AstroGuard</h1>
                <p className="text-xs text-muted-foreground -mt-1">Orbital Sight</p>
              </div>
            </motion.button>

            {/* Desktop Navigation */}
            <div className="hidden lg:flex items-center space-x-6 xl:space-x-8">
              {navItems.map((item) => (
                <motion.button
                  key={item.label}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-300 ${
                    activeSection === item.href.slice(1)
                      ? 'text-primary bg-primary/10'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                  }`}
                  onClick={() => scrollToSection(item.href)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <item.icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                </motion.button>
              ))}
            </div>

            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="sm"
              className="lg:hidden"
              onClick={() => setIsOpen(!isOpen)}
            >
              {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Navigation */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 z-40 lg:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Backdrop */}
            <motion.div
              className="absolute inset-0 bg-background/80 backdrop-blur-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
            />

            {/* Menu Panel */}
            <motion.div
              className="absolute top-16 left-0 right-0 glass border-b border-border/50"
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -20, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <div className="container mx-auto px-4 py-6">
                <div className="space-y-4">
                  {navItems.map((item, index) => (
                    <motion.button
                      key={item.label}
                      className={`flex items-center space-x-3 w-full p-3 rounded-lg transition-all duration-300 ${
                        activeSection === item.href.slice(1)
                          ? 'text-primary bg-primary/10'
                          : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                      }`}
                      onClick={() => scrollToSection(item.href)}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                    >
                      <item.icon className="w-5 h-5" />
                      <span className="font-medium">{item.label}</span>
                    </motion.button>
                  ))}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Action Button */}
      <motion.div
        className="fixed bottom-6 right-6 z-30"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.5, delay: 1 }}
      >
        <Button
          size="lg"
          className="rounded-full cosmic-glow hover:aurora-glow transition-all duration-300 shadow-lg hidden sm:flex"
          onClick={() => scrollToSection('detection')}
        >
          <Target className="w-5 h-5 sm:mr-2" />
          <span className="hidden sm:inline">Try Detection</span>
        </Button>
        {/* Mobile FAB */}
        <Button
          size="sm"
          className="rounded-full cosmic-glow hover:aurora-glow transition-all duration-300 shadow-lg sm:hidden"
          onClick={() => scrollToSection('detection')}
        >
          <Target className="w-4 h-4" />
        </Button>
      </motion.div>

      {/* Progress Indicator */}
      <div className="fixed top-16 left-0 right-0 h-1 bg-border/20 z-40">
        <div 
          className="scroll-progress h-full bg-gradient-to-r from-primary to-accent origin-left transition-transform duration-100 ease-out"
          style={{ transform: 'scaleX(0)' }}
        />
      </div>
    </>
  );
}