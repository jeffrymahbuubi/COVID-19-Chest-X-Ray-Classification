%================================================================================================
% Function: Training the deep learning model & Do Evaluation of the Network Trained
%================================================================================================


load 'Y.mat' %The label of every class, needed for cross validation, in forms of 1x4035 categorical data.
datapath = fullfile(['C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided ' ...
    'Engineering\Quiz, Mid-Term, Final-Project\Final Project\Group 1\Dataset\Chest X-Ray Dataset']); % To locate the DataSet
ResultPath = ['C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided Engineering\Quiz, ' ...
    'Mid-Term, Final-Project\Final Project\Group 1\Training Network Code']; %To locate the save file after training
ResultFolderName = fullfile(ResultPath,'Result'); %To locate the save file after training

X = imageDatastore(datapath,'IncludeSubfolders',true,'LabelSource','foldernames'); %Image Data Store of the Chest X-Ray Dataset
Y = Y; %#ok<ASGSL> 
Model = 'ResNet18'; 
cv = 10; %Value of 'k' in the Cross Validation 
name_class = {'NORMAL','COVID','VIRAL PNEUMONIA'};

rng(1);
t0 = clock;
num_class = size(unique(Y),1);
labels_content = unique(sort(Y));

%% (1) K-fold cross-validation.
if isnumeric(cv)
    x_axis = cell(num_class,cv);	y_axis = cell(num_class,cv);	auc = zeros(num_class,cv);
    x_axis_ave = cell(num_class,1); y_axis_ave = cell(num_class,1);
    each_curve_size = zeros(num_class,cv);
    cm_array = zeros(num_class, num_class, cv);
    accuracy_array = zeros(cv,1);
    sensitivity_array = zeros(num_class,cv);
    specificity_array = zeros(num_class,cv);
    indices = crossvalind('Kfold',Y,cv);
    pre_acc = 0;    Best_Model = [];
    for i = 1:cv
        idxTest = (indices == i);
        idxTrn = ~idxTest;
        
        % Train a model using training set.
        arrTest = imageDatastore(X.Files(idxTest),'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        arrTrn = imageDatastore(X.Files(idxTrn),'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        [Mdl, labels, score] = DL_trainResNet18(Model, arrTrn, arrTest); %Function to train the network
 
        [cm, ~] = confusionmat(Y(idxTest), labels);

        %% (2) ROC curve data
        for n = 1:num_class
            [x_axis{n,i},y_axis{n,i},~,~] = perfcurve(Y(idxTest,:), score(:,n), labels_content(n));
            auc(n,i) = trapz(x_axis{n,i},y_axis{n,i});

            each_curve_size(n,i) = size(x_axis{n,i},1);
            sensitivity_array(n,i) = cm(n,n)/sum(cm(n,:));
            specificity_array(n,i) = (sum(cm, 'all')-sum(cm(n,:))-sum(cm(:,n))+cm(n,n))/(sum(cm, 'all')-sum(cm(n,:)));
        end
        %% (3) Accuracy calculation
        cm_array(:,:,i) = cm;
        accuracy_array(i) = trace(cm)/sum(cm, 'all');
        
        if (accuracy_array(i) > pre_acc)
            pre_acc = accuracy_array(i);
            Best_Model = Mdl;               % save the best model
        end
        
        fprintf('[%s] [%d-fold] completed rate: %d/%d (%g%%)\n', Model, cv, i, cv, i/cv*100);
        fprintf('Time: %g\n', etime(clock,t0));
    end
    %% (4) Resample data point, let each curve size be the same.
    for p = 1:num_class
        %want_xaxis = 0:1/(min_curve_size(p)-1):1;
        want_xaxis = 0:1/(100-1):1;
        for m = 1:cv
            x = cell2mat(cat(1,x_axis(p,m)));
            y = cell2mat(cat(1,y_axis(p,m)));
            [x, index] = unique(x);
            y_axis{p,m} = interp1(x,y(index),want_xaxis','next');
            x_axis{p,m} = want_xaxis';
        end
        x_axis_ave{p} = mean(cell2mat(cat(1,x_axis(p,:))),2);
        y_axis_ave{p} = mean(cell2mat(cat(1,y_axis(p,:))),2);
    end
    %% (5) Use heatmap to draw confusion matrix.
    H = zeros(num_class, num_class);
    cm_array_sum = sum(cm_array,3);
    for index = 1:num_class
        H(index,:) = cm_array_sum(index,:)/sum(cm_array_sum(index,:))*100;
    end
    
    figure
    plotConfMat(cm_array_sum, name_class);
    colorbar;
    caxis([0 100]);
    
    % go back the previous folder and create a new folder.
    % cd ../;
    cd(ResultPath);
    mkdir(ResultFolderName);
    cd(ResultFolderName);
    saveas(gcf,[ResultFolderName,'_',Model,'_cm','.png']);
    close all
    %% (6) Output.
    cm_array = H;
    sensitivity_array = mean(sensitivity_array, 2);
    specificity_array = mean(specificity_array, 2);
    table_ass = [trace(cm_array_sum)/sum(cm_array_sum,'all') mean(sensitivity_array,1) mean(specificity_array,1)];
    writematrix(table_ass, 'ResNet18_SegmentedScore.xlsx'); %Write the table_ass variable into excel ta
    %% (7) ROC curve
        figure
        if eq(num_class, 3)
         x_1 = x_axis_ave{1};    y_1 = y_axis_ave{1};    plot(x_1(1:end), smooth(y_1(1:end)), '-', 'LineWidth', 1.5);	hold on
         x_2 = x_axis_ave{2};    y_2 = y_axis_ave{2};    plot(x_2(1:end), smooth(y_2(1:end)), '--', 'LineWidth', 1.5);    hold on
         x_3 = x_axis_ave{3};    y_3 = y_axis_ave{3};    plot(x_3(1:end), smooth(y_3(1:end)), '-.', 'LineWidth', 1.5);    hold on

         axis([0 1 0 1]);
         xlabel('False positive rate', 'FontSize', 14)
         ylabel('True positive rate', 'FontSize', 14)
         title('ROC curve for Classification', 'FontSize', 14)
         lgd = legend(   [name_class{1}, ', AUC = ', num2str(mean(auc(1,:),'all'))], ...
                [name_class{2}, ', AUC = ', num2str(mean(auc(2,:),'all'))], ...
                [name_class{3}, ', AUC = ', num2str(mean(auc(3,:),'all'))]);
       
        lgd.FontSize = 8;
        lgd.Title.String = Model;
        saveas(gcf,[Model,'_roc','.png']);
        close all
        end
 
end

spend_time = etime(clock, t0);
%% END, don't delete.
cd ../;