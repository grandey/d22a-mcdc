CDF   �   
      time       bnds      lon       lat          2   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �CMCC-ESM2 (2017): 
aerosol: MAM3
atmos: CAM5.3 (1deg; 288 x 192 longitude/latitude; 30 levels; top at ~2 hPa)
atmosChem: none
land: CLM4.5 (BGC mode)
landIce: none
ocean: NEMO3.6 (ORCA1 tripolar primarly 1 deg lat/lon with meridional refinement down to 1/3 degree in the tropics; 362 x 292 longitude/latitude; 50 vertical levels; top grid cell 0-1 m)
ocnBgchem: BFM5.1
seaIce: CICE4.0   institution       QFondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, Lecce 73100, Italy      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    comment       none   contact       	T. Lovato      creation_date         2020-12-25T16:35:41Z   data_specs_version        01.00.31   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.CMCC.CMCC-ESM2.historical.none.r1i1p1f1   grid      native ocean curvilinear grid      
grid_label        gn     history      ]Tue May 30 16:59:06 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CMCC-ESM2_r1i1p1f1/hfds_CMCC-ESM2_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CMCC-ESM2_r1i1p1f1/CMIP6.ScenarioMIP.CMCC.CMCC-ESM2.ssp126.r1i1p1f1.Omon.hfds.gn.v20210126/hfds_Omon_CMCC-ESM2_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacello_ssp126_v20210126.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CMCC-ESM2_r1i1p1f1/hfds_CMCC-ESM2_r1i1p1f1_ssp126.mergetime.nc
Tue May 30 16:59:06 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.Omon.hfds.gn.v20210114/hfds_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20210114.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CMCC-ESM2_r1i1p1f1/hfds_CMCC-ESM2_r1i1p1f1_historical.mergetime.nc
Thu Nov 03 22:27:47 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.Omon.hfds.gn.v20210114/hfds_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacello/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.Ofx.areacello.gn.v20210114/areacello_Ofx_CMCC-ESM2_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.historical.r1i1p1f1.Omon.hfds.gn.v20210114/hfds_Omon_CMCC-ESM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20210114.fldsum.nc
2020-12-25T16:35:41Z ;rewrote data to be consistent with CMIP for variable hfds found in table Omon.;
none      initialization_index            institution_id        CMCC   mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      	CMCC-ESM2      parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      
references        none   run_variant       1st realization    	source_id         	CMCC-ESM2      source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ECreation Date:(05 February 2020) MD5:6a248fd76c55aa6d6f7b3cc6866b5faf      title         #CMCC-ESM2 output prepared for CMIP6    variable_id       hfds   variant_label         r1i1p1f1   license      ?CMIP6 model data produced by CMCC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.6.0      tracking_id       1hdl:21.14100/c52a315c-4b0a-49bb-b4e5-dcc0cf209c1f      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T                   	time_bnds                                 (   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X                  lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y                  hfds                      standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello             8                Aq���   Aq��P   Aq�P   W�Aq�6�   Aq�P   Aq��P   U��Aq���   Aq��P   Aq��P   W,Aq��   Aq��P   Aq�dP   W���Aq���   Aq�dP   Aq��P   Wހ�Aq���   Aq��P   Aq�FP   W��Aq�k�   Aq�FP   Aq��P   V�w?Aq���   Aq��P   Aq�(P   WK��Aq�M�   Aq�(P   Aq��P   W#iAq���   Aq��P   Aq�
P   WY#<Aq�/�   Aq�
P   Aq�{P   T��Aq���   Aq�{P   Aq��P   T`�Aq��   Aq��P   Aq�]P   �4��AqĂ�   Aq�]P   Aq��P   U���Aq���   Aq��P   Aq�?P   W�5Aq�d�   Aq�?P   Aq˰P   W�qAq���   Aq˰P   Aq�!P   ԑ��Aq�F�   Aq�!P   AqВP   ֳٙAqз�   AqВP   Aq�P   �C��Aq�(�   Aq�P   Aq�tP   W���Aqՙ�   Aq�tP   Aq��P   W�^5Aq�
�   Aq��P   Aq�VP   V��Aq�{�   Aq�VP   Aq��P   V��Aq���   Aq��P   Aq�8P   Vo�RAq�]�   Aq�8P   Aq�P   UO7�Aq���   Aq�P   Aq�P   W�� Aq�?�   Aq�P   Aq�P   W�U�Aq��   Aq�P   Aq��P   W=�Aq�!�   Aq��P   Aq�mP   V���Aq��   Aq�mP   Aq��P   W@a�Aq��   Aq��P   Aq�OP   WOvtAq�t�   Aq�OP   Aq��P   V�ΚAq���   Aq��P   Aq�1P   V�S]Aq�V�   Aq�1P   Aq��P   ��Q�Aq���   Aq��P   Aq�P   ��Aq�8�   Aq�P   Aq��P   W��sAq���   Aq��P   Aq��P   W�)�Aq��   Aq��P   ArfP   W��Ar��   ArfP   Ar�P   V� qAr��   Ar�P   ArHP   V6$�Arm�   ArHP   Ar�P   W\��Ar��   Ar�P   Ar*P   W0��ArO�   Ar*P   Ar�P   W�Ar��   Ar�P   ArP   W��Ar1�   ArP   Ar}P   X��Ar��   Ar}P   Ar�P   WۓAr�   Ar�P   Ar_P   U��qAr��   Ar_P   Ar�P   �`��Ar��   Ar�P   ArAP   շ��Arf�   ArAP   Ar�P   W�,gAr��   Ar�P   Ar!#P   W��MAr!H�   Ar!#P   Ar#�P   W8Ar#��   Ar#�P   Ar&P   �Y� Ar&*�   Ar&P   Ar(vP   ���Ar(��   Ar(vP   Ar*�P   WC[>Ar+�   Ar*�P   Ar-XP   W�>�Ar-}�   Ar-XP   Ar/�P   Vm�sAr/��   Ar/�P   Ar2:P   Uy�vAr2_�   Ar2:P   Ar4�P   W�%�Ar4��   Ar4�P   Ar7P   W%M(Ar7A�   Ar7P   Ar9�P   WcN�Ar9��   Ar9�P   Ar;�P   W��Ar<#�   Ar;�P   Ar>oP   �h�EAr>��   Ar>oP   Ar@�P   �g �ArA�   Ar@�P   ArCQP   W�	�ArCv�   ArCQP   ArE�P   W�
�ArE��   ArE�P   ArH3P   W��ArHX�   ArH3P   ArJ�P   WFm�ArJ��   ArJ�P   ArMP   W�!�ArM:�   ArMP   ArO�P   W�x~ArO��   ArO�P   ArQ�P   ֙wVArR�   ArQ�P   ArThP   ��͠ArT��   ArThP   ArV�P   W�K�ArV��   ArV�P   ArYJP   W��ArYo�   ArYJP   Ar[�P   XC�Ar[��   Ar[�P   Ar^,P   W�կAr^Q�   Ar^,P   Ar`�P   W�!Ar`��   Ar`�P   ArcP   V�_xArc3�   ArcP   AreP   V���Are��   AreP   Arg�P   W���Arh�   Arg�P   ArjaP   W���Arj��   ArjaP   Arl�P   W'�Arl��   Arl�P   AroCP   וe�Aroh�   AroCP   Arq�P   W�̂Arq��   Arq�P   Art%P   W��ArtJ�   Art%P   Arv�P   We��Arv��   Arv�P   AryP   V�<XAry,�   AryP   Ar{xP   WCD�Ar{��   Ar{xP   Ar}�P   XL�Ar~�   Ar}�P   Ar�ZP   WaP$Ar��   Ar�ZP   Ar��P   �r�PAr���   Ar��P   Ar�<P   �@5Ar�a�   Ar�<P   Ar��P   W�|vAr���   Ar��P   Ar�P   W��,Ar�C�   Ar�P   Ar��P   W�ojAr���   Ar��P   Ar� P   W��Ar�%�   Ar� P   Ar�qP   W*DaAr���   Ar�qP   Ar��P   �s�Ar��   Ar��P   Ar�SP   WȞAr�x�   Ar�SP   Ar��P   W�J9Ar���   Ar��P   Ar�5P   S�oJAr�Z�   Ar�5P   Ar��P   WџxAr���   Ar��P   Ar�P   WE=�Ar�<�   Ar�P   Ar��P   W��Ar���   Ar��P   Ar��P   WF%�Ar��   Ar��P   Ar�jP   W|�Ar���   Ar�jP   Ar��P   W���Ar� �   Ar��P   Ar�LP   W� :Ar�q�   Ar�LP   Ar��P   V��'Ar���   Ar��P   Ar�.P   W�g�Ar�S�   Ar�.P   Ar��P   W��Ar���   Ar��P   Ar�P   WW�Ar�5�   Ar�P   Ar��P   �&��Ar���   Ar��P   Ar��P   ���Ar��   Ar��P   Ar�cP   W&s�Ar���   Ar�cP   Ar��P   W���Ar���   Ar��P   Ar�EP   W���Ar�j�   Ar�EP   ArĶP   V@ЗAr���   ArĶP   Ar�'P   �W��Ar�L�   Ar�'P   ArɘP   Xl�Arɽ�   ArɘP   Ar�	P   W�� Ar�.�   Ar�	P   Ar�zP   W���ArΟ�   Ar�zP   Ar��P   Vj��Ar��   Ar��P   Ar�\P   ՚�BArӁ�   Ar�\P   Ar��P   WS|TAr���   Ar��P   Ar�>P   W�ncAr�c�   Ar�>P   ArگP   W���Ar���   ArگP   Ar� P   W���Ar�E�   Ar� P   ArߑP   W��Ar߶�   ArߑP   Ar�P   W�_�Ar�'�   Ar�P   Ar�sP   V�vhAr��   Ar�sP   Ar��P   W�QAr�	�   Ar��P   Ar�UP   WO�Ar�z�   Ar�UP   Ar��P   ���Ar���   Ar��P   Ar�7P   Ո�3Ar�\�   Ar�7P   Ar�P   W��iAr���   Ar�P   Ar�P   W���Ar�>�   Ar�P   Ar��P   Wt{Ar���   Ar��P   Ar��P   W��%Ar� �   Ar��P   Ar�lP   W_;�Ar���   Ar�lP   Ar��P   �9�ZAr��   Ar��P   Ar�NP   U��XAr�s�   Ar�NP   As�P   ����As��   As�P   As0P   V��.AsU�   As0P   As�P   X�As��   As�P   As	P   XM��As	7�   As	P   As�P   X>��As��   As�P   As�P   W��As�   As�P   AseP   W]�As��   AseP   As�P   X'��As��   As�P   AsGP   W��mAsl�   AsGP   As�P   W�#As��   As�P   As)P   W%�1AsN�   As)P   As�P   W�a�As��   As�P   AsP   XLԱAs0�   AsP   As!|P   X3��As!��   As!|P   As#�P   W�csAs$�   As#�P   As&^P   W�\DAs&��   As&^P   As(�P   W��As(��   As(�P   As+@P   X)��As+e�   As+@P   As-�P   W���As-��   As-�P   As0"P   W���As0G�   As0"P   As2�P   W�G�As2��   As2�P   As5P   W��0As5)�   As5P   As7uP   X+�@As7��   As7uP   As9�P   X7��As:�   As9�P   As<WP   W���As<|�   As<WP   As>�P   W�w^As>��   As>�P   AsA9P   W���AsA^�   AsA9P   AsC�P   W��,AsC��   AsC�P   AsFP   U���AsF@�   AsFP   AsH�P   X,�AsH��   AsH�P   AsJ�P   Xl��AsK"�   AsJ�P   AsMnP   X,�#AsM��   AsMnP   AsO�P   W�[�AsP�   AsO�P   AsRPP   W���AsRu�   AsRPP   AsT�P   X&�AsT��   AsT�P   AsW2P   XN�AsWW�   AsW2P   AsY�P   W��AsY��   AsY�P   As\P   V�o!As\9�   As\P   As^�P   XR��As^��   As^�P   As`�P   XxڣAsa�   As`�P   AscgP   XO9�Asc��   AscgP   Ase�P   X��Ase��   Ase�P   AshIP   W�.�Ashn�   AshIP   Asj�P   XX�fAsj��   Asj�P   Asm+P   XA��AsmP�   Asm+P   Aso�P   W��oAso��   Aso�P   AsrP   W�}~Asr2�   AsrP   Ast~P   XI�6Ast��   Ast~P   Asv�P   X2u6Asw�   Asv�P   Asy`P   Wܵ�Asy��   Asy`P   As{�P   Wσ�As{��   As{�P   As~BP   XKn3As~g�   As~BP   As��P   XmJ�As���   As��P   As�$P   X��As�I�   As�$P   As��P   WeUFAs���   As��P   As�P   X2BcAs�+�   As�P   As�wP   XJ�4As���   As�wP   As��P   X%�?As��   As��P   As�YP   V�y�As�~�   As�YP   As��P   XH��As���   As��P   As�;P   XENVAs�`�   As�;P   As��P   W��As���   As��P   As�P   W��qAs�B�   As�P   As��P   Xn,;As���   As��P   As��P   X`ޟAs�$�   As��P   As�pP   X\D�As���   As�pP   As��P   W���As��   As��P   As�RP   W�8�As�w�   As�RP   As��P   W�'As���   As��P   As�4P   Xv��As�Y�   As�4P   As��P   X2sAs���   As��P   As�P   W�f�As�;�   As�P   As��P   W�'As���   As��P   As��P   X	��As��   As��P   As�iP   W���As���   As�iP   As��P   X �xAs���   As��P   As�KP   W�=#As�p�   As�KP   As��P   X!%As���   As��P   As�-P   X��As�R�   As�-P   AsP   X'�As���   AsP   As�P   W�GYAs�4�   As�P   AsǀP   WD�kAsǥ�   AsǀP   As��P   X.U�As��   As��P   As�bP   XNx�Aṡ�   As�bP   As��P   W΍�As���   As��P   As�DP   Wr|As�i�   As�DP   AsӵP   W��As���   AsӵP   As�&P   Wڊ�As�K�   As�&P   AsؗP   X�
Asؼ�   AsؗP   As�P   X2߲As�-�   As�P   As�yP   WʝWAsݞ�   As�yP   As��P   Wq!�As��   As��P   As�[P   X6As��   As�[P   As��P   XH>\As���   As��P   As�=P   W�E�As�b�   As�=P   As�P   W�"_As���   As�P   As�P   X7As�D�   As�P   As�P   X%��As��   As�P   As�P   V�^�As�&�   As�P   As�rP   V�,NAs��   As�rP   As��P   X&��As��   As��P   As�TP   X7��As�y�   As�TP   As��P   X��As���   As��P   As�6P   X˔As�[�   As�6P   As��P   W���As���   As��P   AtP   WvZ�At=�   AtP   At�P   X �RAt��   At�P   At�P   XLG�At�   At�P   At	kP   Wŷ�