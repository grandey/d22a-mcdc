CDF   �   
      time       bnds      lon       lat          0   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �UKESM1.0-LL (2018): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: UKCA-StratTrop
land: JULES-ES-1.0
landIce: none
ocean: NEMO-HadGEM3-GO6.0 (eORCA1 tripolar primarily 1 deg with meridional refinement down to 1/3 degree in the tropics; 360 x 330 longitude/latitude; 75 levels; top grid cell 0-1 m)
ocnBgchem: MEDUSA2
seaIce: CICE-HadGEM3-GSI8 (eORCA1 tripolar primarily 1 deg; 360 x 330 longitude/latitude)   institution       BMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK     activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A�        creation_date         2019-06-24T12:29:25Z   
cv_version        6.2.20.1   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Nhttps://furtherinfo.es-doc.org/CMIP6.MOHC.UKESM1-0-LL.historical.none.r1i1p1f2     grid      -Native N96 grid; 192 x 144 longitude/latitude      
grid_label        gn     history      =Wed Aug 10 15:19:34 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/UKESM1-0-LL_r1i1p1f2/rsut_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp585.r1i1p1f2.Amon.rsut.gn.v20190726/rsut_Amon_UKESM1-0-LL_ssp585_r1i1p1f2_gn_201501-204912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp585.r1i1p1f2.Amon.rsut.gn.v20190726/rsut_Amon_UKESM1-0-LL_ssp585_r1i1p1f2_gn_205001-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/UKESM1-0-LL_r1i1p1f2/rsut_UKESM1-0-LL_r1i1p1f2_ssp585.mergetime.nc
Wed Aug 10 15:19:33 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/UKESM1-0-LL_r1i1p1f2/rsut_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc
Fri Apr 08 10:39:44 2022: cdo -O -s -selname,rsut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 10:39:40 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsut.gn.v20190627/rsut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc
2019-06-24T12:18:04Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
2019-06-24T12:17:53Z MIP Convert v1.1.0, Python v2.7.12, Iris v1.13.0, Numpy v1.13.3, netcdftime v1.4.1.      initialization_index            institution_id        MOHC   mip_era       CMIP6      mo_runid      u-bc179    nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      UKESM1-0-LL    parent_time_units         days since 1850-01-01-00-00-00     parent_variant_label      r1i1p1f2   physics_index               product       model-output   realization_index               realm         atmos      	source_id         UKESM1-0-LL    source_type       AOGCM AER BGC CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(13 December 2018) MD5:2b12b5db6db112aa8b8b0d6c1645b121      title         %UKESM1-0-LL output prepared for CMIP6      variable_id       rsut   variant_label         r1i1p1f2   license      XCMIP6 model data produced by the Met Office Hadley Centre is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https://ukesm.ac.uk/cmip6. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   cmor_version      3.4.0      tracking_id       1hdl:21.14100/262e3690-fad7-4892-9b1b-a26f6034ab1d      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T                    	time_bnds                                     lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   
   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment      umo: For instantaneous outputs, this diagnostic represents an average over the radiation time step using the state of the atmosphere (T,q,clouds) from the first dynamics step within that interval. The time coordinate is the start of the radiation time step interval, so the value for t(N) is the average from t(N) to t(N+1)., CMIP_table_comment: at the top of the atmosphere      original_name         $mo: (stash: m01s01i208, lbproc: 128)   cell_measures         area: areacella    history       u2019-06-24T12:29:25Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).                                Aq���   Aq��P   Aq�P   B�`
Aq�6�   Aq�P   Aq��P   BƸ:Aq���   Aq��P   Aq��P   B�P�Aq��   Aq��P   Aq�dP   BŬ�Aq���   Aq�dP   Aq��P   B��Aq���   Aq��P   Aq�FP   B��Aq�k�   Aq�FP   Aq��P   BơAq���   Aq��P   Aq�(P   B�UAq�M�   Aq�(P   Aq��P   BŻtAq���   Aq��P   Aq�
P   B�t�Aq�/�   Aq�
P   Aq�{P   B��Aq���   Aq�{P   Aq��P   B��Aq��   Aq��P   Aq�]P   B�2AqĂ�   Aq�]P   Aq��P   Bơ!Aq���   Aq��P   Aq�?P   Bǁ�Aq�d�   Aq�?P   Aq˰P   B�GAq���   Aq˰P   Aq�!P   BƍnAq�F�   Aq�!P   AqВP   B�NtAqз�   AqВP   Aq�P   BǅLAq�(�   Aq�P   Aq�tP   B���Aqՙ�   Aq�tP   Aq��P   B���Aq�
�   Aq��P   Aq�VP   Bƿ{Aq�{�   Aq�VP   Aq��P   BƉ;Aq���   Aq��P   Aq�8P   B��Aq�]�   Aq�8P   Aq�P   B���Aq���   Aq�P   Aq�P   B�(tAq�?�   Aq�P   Aq�P   B�E�Aq��   Aq�P   Aq��P   Bƣ'Aq�!�   Aq��P   Aq�mP   B�|�Aq��   Aq�mP   Aq��P   B�@�Aq��   Aq��P   Aq�OP   B���Aq�t�   Aq�OP   Aq��P   Bǒ�Aq���   Aq��P   Aq�1P   B�`Aq�V�   Aq�1P   Aq��P   B���Aq���   Aq��P   Aq�P   B�ǹAq�8�   Aq�P   Aq��P   BɃ�Aq���   Aq��P   Aq��P   B���Aq��   Aq��P   ArfP   B�@#Ar��   ArfP   Ar�P   B�DAr��   Ar�P   ArHP   B��Arm�   ArHP   Ar�P   BƳ�Ar��   Ar�P   Ar*P   B�)�ArO�   Ar*P   Ar�P   B�]CAr��   Ar�P   ArP   Bǖ�Ar1�   ArP   Ar}P   B���Ar��   Ar}P   Ar�P   BƑ�Ar�   Ar�P   Ar_P   B���Ar��   Ar_P   Ar�P   B�+�Ar��   Ar�P   ArAP   B�%Arf�   ArAP   Ar�P   B�v�Ar��   Ar�P   Ar!#P   B�KyAr!H�   Ar!#P   Ar#�P   Bǩ�Ar#��   Ar#�P   Ar&P   BǎEAr&*�   Ar&P   Ar(vP   B�#�Ar(��   Ar(vP   Ar*�P   BȂAr+�   Ar*�P   Ar-XP   BȈ�Ar-}�   Ar-XP   Ar/�P   B�:�Ar/��   Ar/�P   Ar2:P   BǬ�Ar2_�   Ar2:P   Ar4�P   BǪlAr4��   Ar4�P   Ar7P   B�z=Ar7A�   Ar7P   Ar9�P   B�|Ar9��   Ar9�P   Ar;�P   B�T6Ar<#�   Ar;�P   Ar>oP   BȜ(Ar>��   Ar>oP   Ar@�P   B�s�ArA�   Ar@�P   ArCQP   B�9ArCv�   ArCQP   ArE�P   B��mArE��   ArE�P   ArH3P   BǮKArHX�   ArH3P   ArJ�P   B��ArJ��   ArJ�P   ArMP   B�b�ArM:�   ArMP   ArO�P   B�l�ArO��   ArO�P   ArQ�P   B�2)ArR�   ArQ�P   ArThP   B��yArT��   ArThP   ArV�P   BǷ�ArV��   ArV�P   ArYJP   B�8ArYo�   ArYJP   Ar[�P   BǑAr[��   Ar[�P   Ar^,P   Bǜ�Ar^Q�   Ar^,P   Ar`�P   BǋAr`��   Ar`�P   ArcP   B�Arc3�   ArcP   AreP   B�O�Are��   AreP   Arg�P   B��Arh�   Arg�P   ArjaP   B�Arj��   ArjaP   Arl�P   B��Arl��   Arl�P   AroCP   B��aAroh�   AroCP   Arq�P   B�:�Arq��   Arq�P   Art%P   B��!ArtJ�   Art%P   Arv�P   Bǲ Arv��   Arv�P   AryP   B���Ary,�   AryP   Ar{xP   BȟAr{��   Ar{xP   Ar}�P   B��Ar~�   Ar}�P   Ar�ZP   B�j�Ar��   Ar�ZP   Ar��P   Bȴ�Ar���   Ar��P   Ar�<P   B�S�Ar�a�   Ar�<P   Ar��P   BȉiAr���   Ar��P   Ar�P   B�t�Ar�C�   Ar�P   Ar��P   B�AAr���   Ar��P   Ar� P   BȽ.Ar�%�   Ar� P   Ar�qP   B��Ar���   Ar�qP   Ar��P   B��Ar��   Ar��P   Ar�SP   B�L�Ar�x�   Ar�SP   Ar��P   BɺRAr���   Ar��P   Ar�5P   B�ݫAr�Z�   Ar�5P   Ar��P   Bȧ�Ar���   Ar��P   Ar�P   B�H�Ar�<�   Ar�P   Ar��P   BȈ�Ar���   Ar��P   Ar��P   B�ՄAr��   Ar��P   Ar�jP   Bȍ�Ar���   Ar�jP   Ar��P   B�$�Ar� �   Ar��P   Ar�LP   B�c�Ar�q�   Ar�LP   Ar��P   BȃAr���   Ar��P   Ar�.P   B�ʫAr�S�   Ar�.P   Ar��P   B��lAr���   Ar��P   Ar�P   B�5�Ar�5�   Ar�P   Ar��P   B��pAr���   Ar��P   Ar��P   B��Ar��   Ar��P   Ar�cP   B�i�Ar���   Ar�cP   Ar��P   B��Ar���   Ar��P   Ar�EP   B�aAr�j�   Ar�EP   ArĶP   B��/Ar���   ArĶP   Ar�'P   B�J�Ar�L�   Ar�'P   ArɘP   B�Arɽ�   ArɘP   Ar�	P   Bʺ�Ar�.�   Ar�	P   Ar�zP   B˚oArΟ�   Ar�zP   Ar��P   B�v�Ar��   Ar��P   Ar�\P   Bʚ!ArӁ�   Ar�\P   Ar��P   B���Ar���   Ar��P   Ar�>P   Bʀ�Ar�c�   Ar�>P   ArگP   B�l�Ar���   ArگP   Ar� P   B�FAr�E�   Ar� P   ArߑP   B�$�Ar߶�   ArߑP   Ar�P   Bʭ�Ar�'�   Ar�P   Ar�sP   Bʮ�Ar��   Ar�sP   Ar��P   B�G�Ar�	�   Ar��P   Ar�UP   B�KAr�z�   Ar�UP   Ar��P   B�Y�Ar���   Ar��P   Ar�7P   B˥nAr�\�   Ar�7P   Ar�P   B�րAr���   Ar�P   Ar�P   B�|DAr�>�   Ar�P   Ar��P   B��Ar���   Ar��P   Ar��P   Bɿ�Ar� �   Ar��P   Ar�lP   B��cAr���   Ar�lP   Ar��P   Bɚ<Ar��   Ar��P   Ar�NP   B̳�Ar�s�   Ar�NP   As�P   B��As��   As�P   As0P   B�Y�AsU�   As0P   As�P   B��As��   As�P   As	P   B��kAs	7�   As	P   As�P   B�zEAs��   As�P   As�P   B�m�As�   As�P   AseP   B��As��   AseP   As�P   BǲEAs��   As�P   AsGP   B���Asl�   AsGP   As�P   BǍXAs��   As�P   As)P   Bǌ�AsN�   As)P   As�P   Bǐ�As��   As�P   AsP   B�vtAs0�   AsP   As!|P   BǮ�As!��   As!|P   As#�P   B��As$�   As#�P   As&^P   BǢ�As&��   As&^P   As(�P   B�4As(��   As(�P   As+@P   BƠ�As+e�   As+@P   As-�P   B���As-��   As-�P   As0"P   B���As0G�   As0"P   As2�P   B��As2��   As2�P   As5P   B��2As5)�   As5P   As7uP   B��As7��   As7uP   As9�P   B���As:�   As9�P   As<WP   B�|~As<|�   As<WP   As>�P   B�7jAs>��   As>�P   AsA9P   Bģ�AsA^�   AsA9P   AsC�P   B�pAsC��   AsC�P   AsFP   B�q�AsF@�   AsFP   AsH�P   B��AsH��   AsH�P   AsJ�P   B�̦AsK"�   AsJ�P   AsMnP   BÃ�AsM��   AsMnP   AsO�P   B�:GAsP�   AsO�P   AsRPP   BæxAsRu�   AsRPP   AsT�P   B��AsT��   AsT�P   AsW2P   B«�AsWW�   AsW2P   AsY�P   BkAsY��   AsY�P   As\P   B�X;As\9�   As\P   As^�P   B�]�As^��   As^�P   As`�P   B���Asa�   As`�P   AscgP   B�4�Asc��   AscgP   Ase�P   B�X>Ase��   Ase�P   AshIP   B���Ashn�   AshIP   Asj�P   B��Asj��   Asj�P   Asm+P   B���AsmP�   Asm+P   Aso�P   B�.�Aso��   Aso�P   AsrP   B��Asr2�   AsrP   Ast~P   B�W�Ast��   Ast~P   Asv�P   B�+HAsw�   Asv�P   Asy`P   B�Asy��   Asy`P   As{�P   B��gAs{��   As{�P   As~BP   B��TAs~g�   As~BP   As��P   B��lAs���   As��P   As�$P   B�*YAs�I�   As�$P   As��P   B�	As���   As��P   As�P   B�;jAs�+�   As�P   As�wP   B��As���   As�wP   As��P   B���As��   As��P   As�YP   B��As�~�   As�YP   As��P   B�_pAs���   As��P   As�;P   B���As�`�   As�;P   As��P   B���As���   As��P   As�P   B��5As�B�   As�P   As��P   B���As���   As��P   As��P   B�*mAs�$�   As��P   As�pP   B��As���   As�pP   As��P   B���As��   As��P   As�RP   B���As�w�   As�RP   As��P   B�&�As���   As��P   As�4P   B��As�Y�   As�4P   As��P   B��As���   As��P   As�P   B�c~As�;�   As�P   As��P   B�9*As���   As��P   As��P   B�@�As��   As��P   As�iP   B�^�As���   As�iP   As��P   B�9�As���   As��P   As�KP   B�-As�p�   As�KP   As��P   B�!�As���   As��P   As�-P   B���As�R�   As�-P   AsP   B�F�As���   AsP   As�P   B��As�4�   As�P   AsǀP   B��Asǥ�   AsǀP   As��P   B�.�As��   As��P   As�bP   B�juAṡ�   As�bP   As��P   B���As���   As��P   As�DP   B��EAs�i�   As�DP   AsӵP   B���As���   AsӵP   As�&P   B�{^As�K�   As�&P   AsؗP   B��Asؼ�   AsؗP   As�P   B��4As�-�   As�P   As�yP   B�g�Asݞ�   As�yP   As��P   B��As��   As��P   As�[P   B��As��   As�[P   As��P   B��yAs���   As��P   As�=P   B��lAs�b�   As�=P   As�P   B�)�As���   As�P   As�P   B�k?As�D�   As�P   As�P   B��As��   As�P   As�P   B��As�&�   As�P   As�rP   B���As��   As�rP   As��P   B���As��   As��P   As�TP   B��DAs�y�   As�TP   As��P   B�r�As���   As��P   As�6P   B�'As�[�   As�6P   As��P   B���As���   As��P   AtP   B��%At=�   AtP   At�P   B��5At��   At�P   At�P   B�ueAt�   At�P   At	kP   B���