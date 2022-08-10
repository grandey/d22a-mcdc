CDF  �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         AJ�       creation_date         2019-02-20T09:32:16Z   data_specs_version        01.00.29   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Khttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.piControl.none.r1i1p1f1    grid      7native atmosphere TL159 gaussian grid (160x320 latxlon)    
grid_label        gn     history      �Wed Aug 10 15:20:32 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/MRI-ESM2-0_r1i1p1f1/rlut_MRI-ESM2-0_r1i1p1f1_piControl.mergetime.nc
Fri Apr 08 07:01:14 2022: cdo -O -s -selname,rlut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 07:00:50 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rlut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.nc
2019-02-20T09:32:16Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
Output from run-Dr060_piControl_100 (sfc_avr_mon.ctl)      initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(14 December 2018) MD5:b2d32d1a0d9b196411429c8895329d8f      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       rlut   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/60ed355a-9500-46dc-947f-b216e7ad5786      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rlut                   
   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      original_name         ULWT   cell_measures         area: areacella    history       r2019-02-20T09:32:16Z altered by CMOR: replaced missing value flag (-9.99e+33) with standard missing value (1e+20).              �                Aq���   Aq��P   Aq�P   Cq��Aq�6�   Aq�P   Aq��P   CqwAq���   Aq��P   Aq��P   Cq��Aq��   Aq��P   Aq�dP   Cq�zAq���   Aq�dP   Aq��P   Cq��Aq���   Aq��P   Aq�FP   CqgAq�k�   Aq�FP   Aq��P   Cq�Aq���   Aq��P   Aq�(P   Cq�Aq�M�   Aq�(P   Aq��P   Cq��Aq���   Aq��P   Aq�
P   CqN�Aq�/�   Aq�
P   Aq�{P   Cq�UAq���   Aq�{P   Aq��P   Cqj�Aq��   Aq��P   Aq�]P   Cqf�AqĂ�   Aq�]P   Aq��P   CqV�Aq���   Aq��P   Aq�?P   Cq]%Aq�d�   Aq�?P   Aq˰P   Cq�JAq���   Aq˰P   Aq�!P   Cr2�Aq�F�   Aq�!P   AqВP   CrbAqз�   AqВP   Aq�P   Cr�Aq�(�   Aq�P   Aq�tP   Cq�NAqՙ�   Aq�tP   Aq��P   Cq��Aq�
�   Aq��P   Aq�VP   Cq��Aq�{�   Aq�VP   Aq��P   Cq}�Aq���   Aq��P   Aq�8P   Cq�`Aq�]�   Aq�8P   Aq�P   Cqw�Aq���   Aq�P   Aq�P   Cq��Aq�?�   Aq�P   Aq�P   Cq�^Aq��   Aq�P   Aq��P   Cqj-Aq�!�   Aq��P   Aq�mP   CqP�Aq��   Aq�mP   Aq��P   Cq�Aq��   Aq��P   Aq�OP   Cq�sAq�t�   Aq�OP   Aq��P   Cq�Aq���   Aq��P   Aq�1P   Cr�Aq�V�   Aq�1P   Aq��P   Cq�Aq���   Aq��P   Aq�P   CqV�Aq�8�   Aq�P   Aq��P   Cq��Aq���   Aq��P   Aq��P   Cq�EAq��   Aq��P   ArfP   Cqy6Ar��   ArfP   Ar�P   Cq�LAr��   Ar�P   ArHP   Cq�uArm�   ArHP   Ar�P   Cq�2Ar��   Ar�P   Ar*P   Cq��ArO�   Ar*P   Ar�P   Cq��Ar��   Ar�P   ArP   Cq��Ar1�   ArP   Ar}P   Cq!�Ar��   Ar}P   Ar�P   Cp�Ar�   Ar�P   Ar_P   Cq�Ar��   Ar_P   Ar�P   CqlAr��   Ar�P   ArAP   Cql�Arf�   ArAP   Ar�P   CqV�Ar��   Ar�P   Ar!#P   CqYVAr!H�   Ar!#P   Ar#�P   CqJ�Ar#��   Ar#�P   Ar&P   Cq�FAr&*�   Ar&P   Ar(vP   Cq��Ar(��   Ar(vP   Ar*�P   Cq��Ar+�   Ar*�P   Ar-XP   Cq�aAr-}�   Ar-XP   Ar/�P   Cq��Ar/��   Ar/�P   Ar2:P   Cq�%Ar2_�   Ar2:P   Ar4�P   Cq?�Ar4��   Ar4�P   Ar7P   Cqp%Ar7A�   Ar7P   Ar9�P   Cq�Ar9��   Ar9�P   Ar;�P   CqG�Ar<#�   Ar;�P   Ar>oP   CqU5Ar>��   Ar>oP   Ar@�P   Cq�oArA�   Ar@�P   ArCQP   Cq{�ArCv�   ArCQP   ArE�P   CqgBArE��   ArE�P   ArH3P   Cq�9ArHX�   ArH3P   ArJ�P   Cq��ArJ��   ArJ�P   ArMP   Cq{JArM:�   ArMP   ArO�P   Cq=ArO��   ArO�P   ArQ�P   Cq�7ArR�   ArQ�P   ArThP   Cq��ArT��   ArThP   ArV�P   Cq{@ArV��   ArV�P   ArYJP   Cq�iArYo�   ArYJP   Ar[�P   CqR�Ar[��   Ar[�P   Ar^,P   Cqd6Ar^Q�   Ar^,P   Ar`�P   Cq��Ar`��   Ar`�P   ArcP   Cq��Arc3�   ArcP   AreP   CqcAre��   AreP   Arg�P   Cq:�Arh�   Arg�P   ArjaP   Cqd�Arj��   ArjaP   Arl�P   CqY�Arl��   Arl�P   AroCP   CqW�Aroh�   AroCP   Arq�P   Cq�dArq��   Arq�P   Art%P   Cq�zArtJ�   Art%P   Arv�P   Cqx�Arv��   Arv�P   AryP   Cq�Ary,�   AryP   Ar{xP   CqS�Ar{��   Ar{xP   Ar}�P   Cq6RAr~�   Ar}�P   Ar�ZP   CqRAr��   Ar�ZP   Ar��P   CqM�Ar���   Ar��P   Ar�<P   Cq�FAr�a�   Ar�<P   Ar��P   CqX�Ar���   Ar��P   Ar�P   Cq}$Ar�C�   Ar�P   Ar��P   Cr�Ar���   Ar��P   Ar� P   CqvAr�%�   Ar� P   Ar�qP   Cqp>Ar���   Ar�qP   Ar��P   Cq�@Ar��   Ar��P   Ar�SP   Cr�Ar�x�   Ar�SP   Ar��P   Cr#Ar���   Ar��P   Ar�5P   Cq|�Ar�Z�   Ar�5P   Ar��P   Cq�Ar���   Ar��P   Ar�P   Cq�cAr�<�   Ar�P   Ar��P   Cq��Ar���   Ar��P   Ar��P   Cq�Ar��   Ar��P   Ar�jP   CqfAr���   Ar�jP   Ar��P   Cq�LAr� �   Ar��P   Ar�LP   Cq�*Ar�q�   Ar�LP   Ar��P   Cqy�Ar���   Ar��P   Ar�.P   Cq��Ar�S�   Ar�.P   Ar��P   Cq>�Ar���   Ar��P   Ar�P   Cq�Ar�5�   Ar�P   Ar��P   Cq�
Ar���   Ar��P   Ar��P   Cqg�Ar��   Ar��P   Ar�cP   Cq�rAr���   Ar�cP   Ar��P   Cqq�Ar���   Ar��P   Ar�EP   Cq�CAr�j�   Ar�EP   ArĶP   Cq�\Ar���   ArĶP   Ar�'P   Cq_eAr�L�   Ar�'P   ArɘP   Cq�cArɽ�   ArɘP   Ar�	P   Cq�uAr�.�   Ar�	P   Ar�zP   Cqa�ArΟ�   Ar�zP   Ar��P   Cq^�Ar��   Ar��P   Ar�\P   Cq��ArӁ�   Ar�\P   Ar��P   Cqp�Ar���   Ar��P   Ar�>P   Cqf�Ar�c�   Ar�>P   ArگP   Cq��Ar���   ArگP   Ar� P   Cq��Ar�E�   Ar� P   ArߑP   CqN�Ar߶�   ArߑP   Ar�P   Cqp%Ar�'�   Ar�P   Ar�sP   Cq��Ar��   Ar�sP   Ar��P   Cq��Ar�	�   Ar��P   Ar�UP   Cq��Ar�z�   Ar�UP   Ar��P   Cq��Ar���   Ar��P   Ar�7P   CqpAr�\�   Ar�7P   Ar�P   Cq�LAr���   Ar�P   Ar�P   Cq�>Ar�>�   Ar�P   Ar��P   Cq�vAr���   Ar��P   Ar��P   Cr-�Ar� �   Ar��P   Ar�lP   Cq��Ar���   Ar�lP   Ar��P   Cq�*Ar��   Ar��P   Ar�NP   Cq��Ar�s�   Ar�NP   As�P   Cq\fAs��   As�P   As0P   Cq�AsU�   As0P   As�P   Cq�	As��   As�P   As	P   Cq�cAs	7�   As	P   As�P   Cq�zAs��   As�P   As�P   CqƵAs�   As�P   AseP   Cq�sAs��   AseP   As�P   Cq��As��   As�P   AsGP   Cq�ZAsl�   AsGP   As�P   Cr%0As��   As�P   As)P   Cq_`AsN�   As)P   As�P   CqgAs��   As�P   AsP   Cq��As0�   AsP   As!|P   Cr+�As!��   As!|P   As#�P   Cq�<As$�   As#�P   As&^P   CrAAs&��   As&^P   As(�P   Cq��As(��   As(�P   As+@P   Cq�TAs+e�   As+@P   As-�P   Cq�As-��   As-�P   As0"P   CqٟAs0G�   As0"P   As2�P   Cq{As2��   As2�P   As5P   Cq�As5)�   As5P   As7uP   Cq~As7��   As7uP   As9�P   Cq�As:�   As9�P   As<WP   Cq�As<|�   As<WP   As>�P   Cq��As>��   As>�P   AsA9P   Cq�HAsA^�   AsA9P   AsC�P   Cq�fAsC��   AsC�P   AsFP   Cq��AsF@�   AsFP   AsH�P   Cq�AsH��   AsH�P   AsJ�P   Cq�AsK"�   AsJ�P   AsMnP   Cq�yAsM��   AsMnP   AsO�P   Cq��AsP�   AsO�P   AsRPP   Cq�^AsRu�   AsRPP   AsT�P   Cq��AsT��   AsT�P   AsW2P   Cq�%AsWW�   AsW2P   AsY�P   Cq[�AsY��   AsY�P   As\P   Cq��As\9�   As\P   As^�P   Cq��As^��   As^�P   As`�P   Cq�2Asa�   As`�P   AscgP   Cq��Asc��   AscgP   Ase�P   Cq��Ase��   Ase�P   AshIP   Cq�rAshn�   AshIP   Asj�P   Cqt�Asj��   Asj�P   Asm+P   Cq^�AsmP�   Asm+P   Aso�P   Cq��Aso��   Aso�P   AsrP   CqԵAsr2�   AsrP   Ast~P   Cq��Ast��   Ast~P   Asv�P   Cq�RAsw�   Asv�P   Asy`P   CqL{Asy��   Asy`P   As{�P   Cq�&As{��   As{�P   As~BP   Cq�_As~g�   As~BP   As��P   Cq�~As���   As��P   As�$P   Cqa�As�I�   As�$P   As��P   Cqi�As���   As��P   As�P   Cr2As�+�   As�P   As�wP   Cq�?As���   As�wP   As��P   Cq�NAs��   As��P   As�YP   CqAs�~�   As�YP   As��P   Cq�xAs���   As��P   As�;P   Cq��As�`�   As�;P   As��P   Cq�ZAs���   As��P   As�P   Cq�ZAs�B�   As�P   As��P   Cq�rAs���   As��P   As��P   Cq]�As�$�   As��P   As�pP   Cq�&As���   As�pP   As��P   Cq�As��   As��P   As�RP   Cqx�As�w�   As�RP   As��P   Cq�(As���   As��P   As�4P   Cq�<As�Y�   As�4P   As��P   Cqd�As���   As��P   As�P   Cq�qAs�;�   As�P   As��P   Cq��As���   As��P   As��P   Cqi�As��   As��P   As�iP   Cq��As���   As�iP   As��P   Cr? As���   As��P   As�KP   Cq�6As�p�   As�KP   As��P   Cqa�As���   As��P   As�-P   Cq��As�R�   As�-P   AsP   Cq��As���   AsP   As�P   Cq�As�4�   As�P   AsǀP   Cp�eAsǥ�   AsǀP   As��P   CqE As��   As��P   As�bP   Cq�aAṡ�   As�bP   As��P   Cq�As���   As��P   As�DP   CqU�As�i�   As�DP   AsӵP   Cq��As���   AsӵP   As�&P   Cq��As�K�   As�&P   AsؗP   Cqf�Asؼ�   AsؗP   As�P   Cq_rAs�-�   As�P   As�yP   Cq��Asݞ�   As�yP   As��P   Cq��As��   As��P   As�[P   Cq�eAs��   As�[P   As��P   Cq{�As���   As��P   As�=P   Cq��As�b�   As�=P   As�P   Cq�NAs���   As�P   As�P   Cq�8As�D�   As�P   As�P   Cq�}As��   As�P   As�P   Cq��As�&�   As�P   As�rP   Cq�hAs��   As�rP   As��P   Cq�0As��   As��P   As�TP   CqfkAs�y�   As�TP   As��P   Cq?As���   As��P   As�6P   Cq�6As�[�   As�6P   As��P   Cq��As���   As��P   AtP   Cq�At=�   AtP   At�P   Cq�At��   At�P   At�P   Cq�At�   At�P   At	kP   CqƲAt	��   At	kP   At�P   Cq�	At�   At�P   AtMP   Cq�PAtr�   AtMP   At�P   Cq�WAt��   At�P   At/P   Cq��AtT�   At/P   At�P   Cq�VAt��   At�P   AtP   Cq��At6�   AtP   At�P   CqtAt��   At�P   At�P   Cq��At�   At�P   AtdP   Cq�BAt��   AtdP   At!�P   Cq��At!��   At!�P   At$FP   CqM�At$k�   At$FP   At&�P   Cq�aAt&��   At&�P   At)(P   Cq��At)M�   At)(P   At+�P   CqVKAt+��   At+�P   At.
P   Cq-!At./�   At.
P   At0{P   Cql�At0��   At0{P   At2�P   Cq��At3�   At2�P   At5]P   Cq��At5��   At5]P   At7�P   Cq��At7��   At7�P   At:?P   Cq��At:d�   At:?P   At<�P   Cr�At<��   At<�P   At?!P   Cq�6At?F�   At?!P   AtA�P   Cq�lAtA��   AtA�P   AtDP   Cq�/AtD(�   AtDP   AtFtP   Cq|AtF��   AtFtP   AtH�P   Cq��AtI
�   AtH�P   AtKVP   Cq�AtK{�   AtKVP   AtM�P   CrAtM��   AtM�P   AtP8P   Cq�AtP]�   AtP8P   AtR�P   Cq��AtR��   AtR�P   AtUP   Cq�MAtU?�   AtUP   AtW�P   Cq�|AtW��   AtW�P   AtY�P   Cr�AtZ!�   AtY�P   At\mP   Cq��At\��   At\mP   At^�P   Cq{�At_�   At^�P   AtaOP   Cq��Atat�   AtaOP   Atc�P   Cr"Atc��   Atc�P   Atf1P   CqG�AtfV�   Atf1P   Ath�P   CqXbAth��   Ath�P   AtkP   Cq·Atk8�   AtkP   Atm�P   Cq|�Atm��   Atm�P   Ato�P   Cq�
Atp�   Ato�P   AtrfP   Cq��Atr��   AtrfP   Att�P   Cqy�Att��   Att�P   AtwHP   Cr3
Atwm�   AtwHP   Aty�P   Cr"�Aty��   Aty�P   At|*P   Cq�}At|O�   At|*P   At~�P   Cq�At~��   At~�P   At�P   Cq��At�1�   At�P   At�}P   Cq�bAt���   At�}P   At��P   Cq��At��   At��P   At�_P   Cq�YAt���   At�_P   At��P   Cq�GAt���   At��P   At�AP   Cq�mAt�f�   At�AP   At��P   Cq�tAt���   At��P   At�#P   Cq�lAt�H�   At�#P   At��P   Cq��At���   At��P   At�P   CqͱAt�*�   At�P   At�vP   Cr4�At���   At�vP   At��P   Cq��At��   At��P   At�XP   Cq��At�}�   At�XP   At��P   Cq��At���   At��P   At�:P   Cq��At�_�   At�:P   At��P   Cq�lAt���   At��P   At�P   Cq}DAt�A�   At�P   At��P   Cq�4At���   At��P   At��P   Cq�'At�#�   At��P   At�oP   Cq��At���   At�oP   At��P   Cq�At��   At��P   At�QP   Cq��At�v�   At�QP   At��P   Cqd�At���   At��P   At�3P   CqwAt�X�   At�3P   At��P   Cq�6At���   At��P   At�P   Cq�dAt�:�   At�P   At��P   Cq�(At���   At��P   At��P   Cq˪At��   At��P   At�hP   Cq��Atō�   At�hP   At��P   Cq;�At���   At��P   At�JP   Cq_At�o�   At�JP   At̻P   Cq�wAt���   At̻P   At�,P   Cq��At�Q�   At�,P   AtѝP   Cqh&At���   AtѝP   At�P   Cq�|At�3�   At�P   At�P   Cq�2At֤�   At�P   At��P   Cq�At��   At��P   At�aP   Cq�Atۆ�   At�aP   At��P   Cq��At���   At��P   At�CP   Cq�uAt�h�   At�CP   At�P   CqV�At���   At�P   At�%P   CqطAt�J�   At�%P   At�P   Cq��At��   At�P   At�P   Cr�At�,�   At�P   At�xP   Cq�?At��   At�xP   At��P   Cq�	At��   At��P   At�ZP   Cq�]At��   At�ZP   At��P   Cq�>At���   At��P   At�<P   Cq�<At�a�   At�<P   At��P   Cq�.At���   At��P   At�P   Cqm/At�C�   At�P   At��P   Cq��At���   At��P   Au  P   Cq�Au %�   Au  P   AuqP   Cqq|Au��   AuqP   Au�P   CrX�Au�   Au�P   AuSP   Cr72Aux�   AuSP   Au	�P   Cq�/Au	��   Au	�P   Au5P   Cql AuZ�   Au5P   Au�P   Cq��Au��   Au�P   AuP   CqF Au<�   AuP   Au�P   Cq0�Au��   Au�P   Au�P   Cq�Au�   Au�P   AujP   Cq��Au��   AujP   Au�P   Cr�Au �   Au�P   AuLP   Cq��Auq�   AuLP   Au�P   Cq��Au��   Au�P   Au".P   Cq��Au"S�   Au".P   Au$�P   Cr!�Au$��   Au$�P   Au'P   Cq�xAu'5�   Au'P   Au)�P   Cq�2Au)��   Au)�P   Au+�P   Cq��Au,�   Au+�P   Au.cP   Cq��Au.��   Au.cP   Au0�P   Cq�?Au0��   Au0�P   Au3EP   Cq��Au3j�   Au3EP   Au5�P   CqnyAu5��   Au5�P   Au8'P   CqV�Au8L�   Au8'P   Au:�P   Cq_�Au:��   Au:�P   Au=	P   Cq��Au=.�   Au=	P   Au?zP   Cq��Au?��   Au?zP   AuA�P   CqL�AuB�   AuA�P   AuD\P   Cq�AuD��   AuD\P   AuF�P   Cq��AuF��   AuF�P   AuI>P   Cq'�AuIc�   AuI>P   AuK�P   CqU_AuK��   AuK�P   AuN P   CqҀAuNE�   AuN P   AuP�P   Cq��AuP��   AuP�P   AuSP   Cr�AuS'�   AuSP   AuUsP   CqچAuU��   AuUsP   AuW�P   CrDDAuX	�   AuW�P   AuZUP   Cq��AuZz�   AuZUP   Au\�P   CquyAu\��   Au\�P   Au_7P   Cq�8Au_\�   Au_7P   Aua�P   Cq��Aua��   Aua�P   AudP   Cq�4Aud>�   AudP   Auf�P   Cq��Auf��   Auf�P   Auh�P   CqņAui �   Auh�P   AuklP   Cq�_Auk��   AuklP   Aum�P   CqʹAun�   Aum�P   AupNP   Cq��Aups�   AupNP   Aur�P   Cq��Aur��   Aur�P   Auu0P   Cq��AuuU�   Auu0P   Auw�P   Cq�Auw��   Auw�P   AuzP   Cr&�Auz7�   AuzP   Au|�P   Cq�uAu|��   Au|�P   Au~�P   Cq�Au�   Au~�P   Au�eP   CqݞAu���   Au�eP   Au��P   CqnaAu���   Au��P   Au�GP   Cq~_Au�l�   Au�GP   Au��P   Cq�LAu���   Au��P   Au�)P   Cq��Au�N�   Au�)P   Au��P   Cq�RAu���   Au��P   Au�P   Cr�Au�0�   Au�P   Au�|P   Cq��Au���   Au�|P   Au��P   Cq�|Au��   Au��P   Au�^P   Cq�_Au���   Au�^P   Au��P   CqѣAu���   Au��P   Au�@P   Cq��Au�e�   Au�@P   Au��P   Cq�EAu���   Au��P   Au�"P   CqɶAu�G�   Au�"P   Au��P   Cq�gAu���   Au��P   Au�P   Cq�rAu�)�   Au�P   Au�uP   Cr5/Au���   Au�uP   Au��P   Cq�uAu��   Au��P   Au�WP   CqR�Au�|�   Au�WP   Au��P   Cq�Au���   Au��P   Au�9P   Cq�dAu�^�   Au�9P   Au��P   Cq�Au���   Au��P   Au�P   Cq��Au�@�   Au�P   Au��P   Cq�JAu���   Au��P   Au��P   Cq۶Au�"�   Au��P   Au�nP   Cq�Au���   Au�nP   Au��P   Cq�Au��   Au��P   Au�PP   Cq��Au�u�   Au�PP   Au��P   Cq��Au���   Au��P   Au�2P   Cq��Au�W�   Au�2P   AuʣP   Cr#MAu���   AuʣP   Au�P   Cq��Au�9�   Au�P   AuυP   Cq��AuϪ�   AuυP   Au��P   Cq�Au��   Au��P   Au�gP   Cqw�AuԌ�   Au�gP   Au��P   Cr�Au���   Au��P   Au�IP   Cr`Au�n�   Au�IP   AuۺP   Cr�#Au���   AuۺP   Au�+P   Cql2Au�P�   Au�+P   Au��P   Cq��Au���   Au��P   Au�P   Cr$�Au�2�   Au�P   Au�~P   Cr
�Au��   Au�~P   Au��P   Cq��Au��   Au��P   Au�`P   Cq�aAu��   Au�`P   Au��P   Cq��Au���   Au��P   Au�BP   CqɐAu�g�   Au�BP   Au�P   CqƛAu���   Au�P   Au�$P   Cq��Au�I�   Au�$P   Au��P   Cq�OAu���   Au��P   Au�P   CqqAu�+�   Au�P   Au�wP   Cqt�Au���   Au�wP   Au��P   Cq�Au��   Au��P   Av YP   Cq�Av ~�   Av YP   Av�P   Cq��Av��   Av�P   Av;P   Cq�Av`�   Av;P   Av�P   Cq��Av��   Av�P   Av
P   Cq{Av
B�   Av
P   Av�P   CrAv��   Av�P   Av�P   CqO�Av$�   Av�P   AvpP   CqYEAv��   AvpP   Av�P   Cq��Av�   Av�P   AvRP   Cq�8Avw�   AvRP   Av�P   CqфAv��   Av�P   Av4P   Cq��AvY�   Av4P   Av�P   Cq��Av��   Av�P   Av P   Cr�Av ;�   Av P   Av"�P   Cqw.Av"��   Av"�P   Av$�P   Cqo�Av%�   Av$�P   Av'iP   Cq�wAv'��   Av'iP   Av)�P   Cq�Av)��   Av)�P   Av,KP   Cr\Av,p�   Av,KP   Av.�P   Cq�zAv.��   Av.�P   Av1-P   Cqa�Av1R�   Av1-P   Av3�P   Cq��Av3��   Av3�P   Av6P   Cq��Av64�   Av6P   Av8�P   Cq�hAv8��   Av8�P   Av:�P   Cq��Av;�   Av:�P   Av=bP   Cq�nAv=��   Av=bP   Av?�P   Cqk�Av?��   Av?�P   AvBDP   CqxuAvBi�   AvBDP   AvD�P   Cq��AvD��   AvD�P   AvG&P   Cq��AvGK�   AvG&P   AvI�P   CqlAAvI��   AvI�P   AvLP   Cq�ZAvL-�   AvLP   AvNyP   Cqt�AvN��   AvNyP   AvP�P   Cq|�AvQ�   AvP�P   AvS[P   Cq��AvS��   AvS[P   AvU�P   CqzDAvU��   AvU�P   AvX=P   Cq�TAvXb�   AvX=P   AvZ�P   CqٙAvZ��   AvZ�P   Av]P   Cq�ZAv]D�   Av]P   Av_�P   Cq�<Av_��   Av_�P   AvbP   CqvAvb&�   AvbP   AvdrP   Cq٢Avd��   AvdrP   Avf�P   Cq��Avg�   Avf�P   AviTP   Cq��Aviy�   AviTP   Avk�P   Cq�EAvk��   Avk�P   Avn6P   Cq��Avn[�   Avn6P   Avp�P   Cq�Avp��   Avp�P   AvsP   Cq�WAvs=�   AvsP   Avu�P   Cq�Avu��   Avu�P   Avw�P   Cq��Avx�   Avw�P   AvzkP   Cq��Avz��   AvzkP   Av|�P   Cq��Av}�   Av|�P   AvMP   Cr*Avr�   AvMP   Av��P   Cq�!Av���   Av��P   Av�/P   Cq��Av�T�   Av�/P   Av��P   Cq�Av���   Av��P   Av�P   Cq��Av�6�   Av�P   Av��P   Cq�
Av���   Av��P   Av��P   CqڋAv��   Av��P   Av�dP   Cq�Av���   Av�dP   Av��P   Cq��Av���   Av��P   Av�FP   Cq��Av�k�   Av�FP   Av��P   Cq��Av���   Av��P   Av�(P   CrAv�M�   Av�(P   Av��P   Cq��Av���   Av��P   Av�
P   Cq�<Av�/�   Av�
P   Av�{P   Cq�xAv���   Av�{P   Av��P   Cr*}Av��   Av��P   Av�]P   Cq�ZAv���   Av�]P   Av��P   CrA�Av���   Av��P   Av�?P   Cq��Av�d�   Av�?P   Av��P   Cq��Av���   Av��P   Av�!P   Cq�Av�F�   Av�!P   Av��P   Cq�-Av���   Av��P   Av�P   Cq�Av�(�   Av�P   Av�tP   Cq��Av���   Av�tP   Av��P   Cq�<Av�
�   Av��P   Av�VP   Cq�?Av�{�   Av�VP   Av��P   Cq��Av���   Av��P   Av�8P   CqёAv�]�   Av�8P   AvéP   Cq�/Av���   AvéP   Av�P   Cq�YAv�?�   Av�P   AvȋP   CqtaAvȰ�   AvȋP   Av��P   Cq��Av�!�   Av��P   Av�mP   Cr	�Av͒�   Av�mP   Av��P   Cq�Av��   Av��P   Av�OP   Cq��Av�t�   Av�OP   Av��P   Cq�xAv���   Av��P   Av�1P   Cr�Av�V�   Av�1P   Av٢P   Cq�Av���   Av٢P   Av�P   Cq��Av�8�   Av�P   AvބP   Cq�wAvީ�   AvބP   Av��P   Cq�RAv��   Av��P   Av�fP   Cq�8Av��   Av�fP   Av��P   Cq��Av���   Av��P   Av�HP   Cq��Av�m�   Av�HP   Av�P   Cq��Av���   Av�P   Av�*P   Cqd^Av�O�   Av�*P   Av�P   Cq��Av���   Av�P   Av�P   Cq�CAv�1�   Av�P   Av�}P   CqfnAv���   Av�}P   Av��P   Cqo	Av��   Av��P   Av�_P   Cq��Av���   Av�_P   Av��P   Cq�xAv���   Av��P   Av�AP   Cq�Av�f�   Av�AP   Aw �P   CqH�Aw ��   Aw �P   Aw#P   Cq�KAwH�   Aw#P   Aw�P   Cq�DAw��   Aw�P   AwP   Cq�yAw*�   AwP   Aw
vP   Cq�;Aw
��   Aw
vP   Aw�P   Cq~Aw�   Aw�P   AwXP   Cq�5Aw}�   AwXP   Aw�P   Cq�bAw��   Aw�P   Aw:P   Cq�Aw_�   Aw:P   Aw�P   CqAw��   Aw�P   AwP   CqӄAwA�   AwP   Aw�P   Cq�Aw��   Aw�P   Aw�P   Cq��Aw#�   Aw�P   Aw oP   Cq��Aw ��   Aw oP   Aw"�P   Cq�hAw#�   Aw"�P   Aw%QP   Cq��Aw%v�   Aw%QP   Aw'�P   Cq��Aw'��   Aw'�P   Aw*3P   Cq�DAw*X�   Aw*3P   Aw,�P   Cq��Aw,��   Aw,�P   Aw/P   Cq��Aw/:�   Aw/P   Aw1�P   Cr�Aw1��   Aw1�P   Aw3�P   Cq�$Aw4�   Aw3�P   Aw6hP   Cq��Aw6��   Aw6hP   Aw8�P   Cr	Aw8��   Aw8�P   Aw;JP   Cq�(Aw;o�   Aw;JP   Aw=�P   Cq�Aw=��   Aw=�P   Aw@,P   Cr�Aw@Q�   Aw@,P   AwB�P   CrVAwB��   AwB�P   AwEP   Cq�PAwE3�   AwEP   AwGP   Cq��AwG��   AwGP   AwI�P   Cqq�AwJ�   AwI�P   AwLaP   Cq�AwL��   AwLaP   AwN�P   Cq�AwN��   AwN�P   AwQCP   Cq��AwQh�   AwQCP   AwS�P   Cq��AwS��   AwS�P   AwV%P   Cq��AwVJ�   AwV%P   AwX�P   Cq��AwX��   AwX�P   Aw[P   CqПAw[,�   Aw[P   Aw]xP   Cq�Aw]��   Aw]xP   Aw_�P   CqıAw`�   Aw_�P   AwbZP   CqԸAwb�   AwbZP   Awd�P   Cq�EAwd��   Awd�P   Awg<P   Cq�TAwga�   Awg<P   Awi�P   Cq�LAwi��   Awi�P   AwlP   Cq�MAwlC�   AwlP   Awn�P   Cq��Awn��   Awn�P   Awq P   Cq��Awq%�   Awq P   AwsqP   Cq��Aws��   AwsqP   Awu�P   Cq�AAwv�   Awu�P   AwxSP   Cq�Awxx�   AwxSP   Awz�P   Cq�Awz��   Awz�P   Aw}5P   Cq��Aw}Z�   Aw}5P   Aw�P   Cq��Aw��   Aw�P   Aw�P   Cq�RAw�<�   Aw�P   Aw��P   Cq�Aw���   Aw��P   Aw��P   Cq��Aw��   Aw��P   Aw�jP   CrHAw���   Aw�jP   Aw��P   Cr#PAw� �   Aw��P   Aw�LP   Cq��Aw�q�   Aw�LP   Aw��P   CrvAw���   Aw��P   Aw�.P   Cq�+Aw�S�   Aw�.P   Aw��P   Cq�Aw���   Aw��P   Aw�P   CqՏAw�5�   Aw�P   Aw��P   Cq�,Aw���   Aw��P   Aw��P   CqϳAw��   Aw��P   Aw�cP   CqheAw���   Aw�cP   Aw��P   Cq��Aw���   Aw��P   Aw�EP   Cq�XAw�j�   Aw�EP   Aw��P   Cq��Aw���   Aw��P   Aw�'P   Cq�/Aw�L�   Aw�'P   Aw��P   Cq��Aw���   Aw��P   Aw�	P   Cq��Aw�.�   Aw�	P   Aw�zP   Cq�BAw���   Aw�zP   Aw��P   CqˏAw��   Aw��P   Aw�\P   Cq�\Aw���   Aw�\P   Aw��P   Cr�Aw���   Aw��P   Aw�>P   Cq�_Aw�c�   Aw�>P   Aw��P   Cq�Aw���   Aw��P   Aw� P   Cq�Aw�E�   Aw� P   Aw��P   Cq�6Aw���   Aw��P   Aw�P   Cq�GAw�'�   Aw�P   Aw�sP   CqmQAwƘ�   Aw�sP   Aw��P   Cq�eAw�	�   Aw��P   Aw�UP   Cq�:Aw�z�   Aw�UP   Aw��P   Cq�0Aw���   Aw��P   Aw�7P   Cq��Aw�\�   Aw�7P   AwҨP   CrLAw���   AwҨP   Aw�P   Cr1Aw�>�   Aw�P   Aw׊P   Cq4�Awׯ�   Aw׊P   Aw��P   Cq��Aw� �   Aw��P   Aw�lP   Cq�Awܑ�   Aw�lP   Aw��P   Cqh�Aw��   Aw��P   Aw�NP   Cq��Aw�s�   Aw�NP   Aw�P   Cq��Aw���   Aw�P   Aw�0P   Cq�Aw�U�   Aw�0P   Aw�P   CqS�Aw���   Aw�P   Aw�P   Cq�Aw�7�   Aw�P   Aw�P   CqAw���   Aw�P   Aw��P   Cq��Aw��   Aw��P   Aw�eP   Cq�Aw��   Aw�eP   Aw��P   Cr�Aw���   Aw��P   Aw�GP   Cq�Aw�l�   Aw�GP   Aw��P   Cq��Aw���   Aw��P   Aw�)P   Cq�Aw�N�   Aw�)P   Aw��P   CqؖAw���   Aw��P   AxP   Cq��Ax0�   AxP   Ax|P   Cqt�Ax��   Ax|P   Ax�P   Cq�cAx�   Ax�P   Ax^P   Cq�gAx��   Ax^P   Ax
�P   Cq��Ax
��   Ax
�P   Ax@P   Cq]Axe�   Ax@P   Ax�P   Cq�9Ax��   Ax�P   Ax"P   CrZAxG�   Ax"P   Ax�P   Cr,�Ax��   Ax�P   AxP   Cqe�Ax)�   AxP   AxuP   Cq��Ax��   AxuP   Ax�P   Cq�%Ax�   Ax�P   AxWP   CqnkAx|�   AxWP   Ax �P   CqoBAx ��   Ax �P   Ax#9P   Cq�Ax#^�   Ax#9P   Ax%�P   Cq�=Ax%��   Ax%�P   Ax(P   Cq��Ax(@�   Ax(P   Ax*�P   Cq��Ax*��   Ax*�P   Ax,�P   CqʚAx-"�   Ax,�P   Ax/nP   Cq�/Ax/��   Ax/nP   Ax1�P   CrWAx2�   Ax1�P   Ax4PP   Cqy)Ax4u�   Ax4PP   Ax6�P   Cq�qAx6��   Ax6�P   Ax92P   Cq��Ax9W�   Ax92P   Ax;�P   Cq�Ax;��   Ax;�P   Ax>P   Cq�VAx>9�   Ax>P   Ax@�P   Cq�sAx@��   Ax@�P   AxB�P   Cr)XAxC�   AxB�P   AxEgP   Cr�AxE��   AxEgP   AxG�P   Cq^�AxG��   AxG�P   AxJIP   Cq� AxJn�   AxJIP   AxL�P   Cq��AxL��   AxL�P   AxO+P   Cq��AxOP�   AxO+P   AxQ�P   Cq�jAxQ��   AxQ�P   AxTP   Cq��