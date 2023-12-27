
#create_df <- function(){
y_true <- read.csv('Biomec/Paper/data/y_true.csv')[,-1]
y_pred <- read.csv('Biomec/Paper/data/y_pred.csv')[,-1]

colnames(y_true) <- c('Fx', 'Fy', 'Fz')
colnames(y_pred) <- c('Fx', 'Fy', 'Fz')

# X0, X1, X2 = Fx, Fy, Fz
head(y_true)

#plot(y_true[, 3], y_pred[, 3])
#abline(a = 0, b = 1, col = 'red')

pdf('Biomec/Paper/forcePred.pdf')
par(mfrow = c(3, 1), oma = c(4, 1, 1, 1), mar = c(0, 0, 0, 0))
n_samples <- 500

for (force in c('Fx', 'Fy', 'Fz')){
     par(mai=c(0,  0.5, 0, 0)) # bottom, left, top, right
     plot(-1, -1, ylim =c(-0.2, 1), xlim = c(0, 500), axes = FALSE, ann = FALSE)
     lines(1:500, y_true[0:500,force], lwd = 2)
     lines(1:500, y_pred[0:500,force], lty = 2, lwd = 2)
     axis(2, at = c(0, 0.5, 1), labels = c(0, 0.5, 1))
     #mtext(paste('Scaled', force, '(N)'), side=2, line=3)
     
     mtext(paste('          Scaled', force, '(N)'), side = 2, line = 2.2, cex = 0.75)
     
}
axis(1, pos = 0)
mtext('Seconds (s)', side = 1, cex = 0.75)


par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = 'l', bty = 'n', yaxt = 'n', xaxt = 'n')


legend('bottom',legend = c("Measured Force", "Predicted Force"), lwd = 2, xpd = TRUE, horiz = TRUE, lty = c(1, 2),  bty = 'n')

dev.off()